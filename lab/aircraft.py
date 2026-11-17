"""JSBSim set-up for one flight.

The aircraft is a copy of JSBSim's shipped B747 with two changes, written to
lab/aircraft/B747P/B747P.xml on first use:
  * a 'payload' point mass at the reference CG (so gross weight can vary)
  * main tanks enlarged to TANK_CAPACITY_LBS (the shipped 10.9 klb tanks are small)
"""
import functools
import re
from pathlib import Path

import jsbsim
import numpy as np

DT_HZ = 60.0              # physics / autopilot rate
LOG_HZ = 4.0              # logging rate
N_ENG = 4
TANK_CAPACITY_LBS = 25000.0
AIRCRAFT = 'B747P'
JSBSIM_ROOT = Path(jsbsim.__file__).parent
AIRCRAFT_DIR = Path(__file__).parent / 'aircraft'

_MISSING = re.compile(r'The property (\S+) does not exist')


def ensure_aircraft():
    dst = AIRCRAFT_DIR / AIRCRAFT / f'{AIRCRAFT}.xml'
    src = (JSBSIM_ROOT / 'aircraft' / 'B747' / 'B747.xml').read_text(encoding='utf-8')
    pm = ('        <pointmass name="payload">\n'
          '            <weight unit="LBS"> 0 </weight>\n'
          '            <location unit="IN"> <x> 1327 </x> <y> 0 </y> <z> -24 </z> </location>\n'
          '        </pointmass>\n    </mass_balance>')
    if src.count('</mass_balance>') != 1:
        raise RuntimeError('unexpected B747.xml layout')
    out = src.replace('    </mass_balance>', pm, 1)
    out = out.replace('<capacity unit="LBS"> 10912.8 </capacity>',
                      f'<capacity unit="LBS"> {TANK_CAPACITY_LBS:.1f} </capacity>')
    if not dst.exists() or dst.read_text(encoding='utf-8') != out:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(out, encoding='utf-8')


def _run_ic(f):
    """Some models raise 'property does not exist' on the first run_ic();
    setting that property to 0 and retrying clears it (a JSBSim quirk)."""
    for _ in range(12):
        try:
            f.run_ic()
            return
        except Exception as e:
            m = _MISSING.search(str(e))
            if not m:
                raise
            f[m.group(1)] = 0.0


@functools.lru_cache(maxsize=1)
def elevator_sign():
    """Measure (never assume) which way a positive elevator command pitches."""
    f = jsbsim.FGFDMExec(None)
    f.set_debug_level(0)
    f.load_model('B747')
    f.set_dt(1.0 / DT_HZ)
    f['ic/h-sl-ft'], f['ic/vc-kts'], f['ic/gamma-deg'], f['ic/psi-true-deg'] = 15000.0, 250.0, 0.0, 90.0
    _run_ic(f)
    f['propulsion/set-running'] = -1
    f['simulation/do_simple_trim'] = 1
    for _ in range(60):
        f.run()
    before = f['attitude/theta-deg']
    f['fcs/elevator-cmd-norm'] = 0.25
    for _ in range(120):
        f.run()
    return 1.0 if f['attitude/theta-deg'] > before else -1.0


def new_fdm(spec):
    """Trimmed B747P at the cruise condition of `spec`, then disturbed."""
    ensure_aircraft()
    f = jsbsim.FGFDMExec(str(JSBSIM_ROOT))
    f.set_debug_level(0)
    f.set_aircraft_path(str(AIRCRAFT_DIR))
    if not f.load_model(AIRCRAFT):
        raise RuntimeError('cannot load B747P')
    f.set_dt(1.0 / DT_HZ)
    f['atmosphere/delta-T'] = spec.isa_dev_c * 1.8                      # deg C -> Rankine
    f['inertia/pointmass-weight-lbs[0]'] = spec.payload_lbs
    for i in range(N_ENG):
        f[f'propulsion/tank[{i}]/contents-lbs'] = TANK_CAPACITY_LBS * spec.fuel_frac[i]
    f['propulsion/tank[4]/contents-lbs'] = spec.center_fuel_lbs
    f['ic/h-sl-ft'] = spec.cruise_alt
    f['ic/vc-kts'] = spec.start_cas
    f['ic/gamma-deg'] = 0.0
    f['ic/psi-true-deg'] = spec.heading
    _run_ic(f)
    f['propulsion/set-running'] = -1
    f['simulation/do_simple_trim'] = 1
    # initial-state disturbance after trim (controls stay trimmed)
    f['ic/theta-deg'] = f['attitude/theta-deg'] + spec.ic_theta_deg
    f['ic/phi-deg'] = spec.ic_phi_deg
    f['ic/beta-deg'] = spec.ic_beta_deg
    f['ic/q-rad_sec'] = spec.ic_q
    f['ic/p-rad_sec'] = spec.ic_p
    f['ic/r-rad_sec'] = spec.ic_r
    f.run_ic()
    return f


def set_wind(f, spec, alt_ft):
    """Steady wind growing with altitude (40 % at the ground) and veering."""
    frac = min(max(alt_ft / spec.cruise_alt, 0.0), 1.0)
    v = spec.wind_kts * (0.4 + 0.6 * frac) * 1.68781                  # kt -> ft/s
    to = np.radians(spec.wind_from_deg + spec.wind_veer_deg * frac + 180.0)
    f['atmosphere/wind-north-fps'] = v * np.cos(to)
    f['atmosphere/wind-east-fps'] = v * np.sin(to)


def enable_turbulence(f, seed, severity, wind20_fps=25.0):
    """JSBSim MIL-SPEC (Dryden) turbulence; severity 0 calm, 3 light, 4 moderate.
    The seed makes the gust sequence reproducible."""
    f['simulation/randomseed'] = int(seed) % (2 ** 31)
    if severity <= 0:
        return
    f['atmosphere/turb-type'] = 3
    f['atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps'] = wind20_fps
    f['atmosphere/turbulence/milspec/severity'] = severity


def egt_c(tat_c, thr_pos, running, stalled):
    """JSBSim FGTurbine's internal EGT law (computed inside JSBSim, not exposed)."""
    if stalled:
        return tat_c + 903.14
    if not running:
        return tat_c
    return tat_c + 363.1 + 357.1 * thr_pos
