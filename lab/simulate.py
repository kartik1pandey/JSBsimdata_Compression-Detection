"""Fly one cruise flight in closed loop and log EVERYTHING at 4 Hz.

One physics step (60 Hz):
    1. flight plan        -> selected altitude / CAS / heading
    2. SENSORS            -> the autopilot's readings  (observation anomaly: sense())
    3. AUTOPILOT          -> commands: elevator, aileron, rudder, throttle
    4. ACTUATORS          -> applied commands           (control anomaly: actuate())
    5. JSBSim step

Logged columns (4 Hz):
    t, label                         time since start of cruise [s]; 1 inside the anomaly window
    sensor/<X>                       what the sensors read (incl. the anomaly) + measurement noise:
                                     ALT CAS VS PITCH ROLL HDG BETA Q P (autopilot inputs), TLA1..4 (levers)
    ap/*                             autopilot outputs (commands BEFORE any control anomaly) and targets
    sel/*                            selected flight-plan values
    derived/engine[i]/egt-degc       JSBSim's internal EGT law (not exposed as a property)
    <every JSBSim property>          the TRUE simulated state: ~720 properties, '[0]' made explicit,
                                     e.g. propulsion/engine[0]/n1, fcs/elevator-pos-deg, aero/alpha-deg
"""
import numpy as np
import pandas as pd

from . import aircraft as ac
from .anomalies import SENSED, PROP_TO_SENSED, make, envelope
from .autopilot import Autopilot

SENSOR_NOISE = {          # sd, in the logged unit (assumed plausible levels)
    'ALT': 10.0, 'CAS': 0.5, 'VS': 30.0, 'PITCH': 0.05, 'ROLL': 0.05, 'HDG': 0.1,
    'BETA': 0.05, 'Q': 0.02, 'P': 0.02, 'TLA': 0.002,
}
# landing gear / ground reaction: meaningless in cruise, and some of these hold
# uninitialised memory (random 1e180 / 1e-312 values) while airborne
SKIP_PREFIXES = ('gear/', 'ground/')
CMD_PROPS = {'elevator': 'fcs/elevator-cmd-norm', 'aileron': 'fcs/aileron-cmd-norm',
             'rudder': 'fcs/rudder-cmd-norm'}


def canonical_names(names):
    """JSBSim lists element 0 of an array without an index ('propulsion/engine/n1')
    and the rest with one ('propulsion/engine[1]/n1'). Make [0] explicit."""
    bases = set()
    for n in names:
        parts = n.split('/')
        for i, p in enumerate(parts):
            if p.endswith(']') and '[' in p and not p.endswith('[0]'):
                bases.add('/'.join(parts[:i] + [p[:p.index('[')]]))
    out = []
    for n in names:
        parts = n.split('/')
        for i in range(len(parts)):
            if '/'.join(parts[:i + 1]) in bases and not parts[i].endswith(']'):
                parts[i] += '[0]'
        out.append('/'.join(parts))
    return out


def property_nodes(fdm):
    raw = [c.split(' (')[0] for c in fdm.get_property_catalog()]
    pm = fdm.get_property_manager()
    names, nodes = [], []
    for raw_name, name in zip(raw, canonical_names(raw)):
        if name.startswith(SKIP_PREFIXES):
            continue
        try:
            nd = pm.get_node(raw_name)
            nd.get_double_value()
        except Exception:
            continue
        names.append(name)
        nodes.append(nd)
    return names, nodes


class Sensed:
    """The autopilot's view: property reads of the 9 sensed variables go through
    the anomaly's sense(); everything else is passed straight to JSBSim."""
    def __init__(self, fdm, anom):
        self.f, self.anom, self.t, self.cache = fdm, anom, 0.0, {}

    def step(self, t):
        self.t = t
        self.cache.clear()

    def __getitem__(self, k):
        name = PROP_TO_SENSED.get(k)
        if name is None:
            return self.f[k]
        if k not in self.cache:
            self.cache[k] = self.anom.sense(self.t, name, self.f[k], self.f)
        return self.cache[k]


def fly(spec, anomaly=None):
    """Return the 4 Hz log of one flight as a DataFrame (before sensor noise)."""
    anom = make(anomaly, spec.seed)
    fdm = ac.new_fdm(spec)
    dt = 1.0 / ac.DT_HZ
    ap = Autopilot(ac.elevator_sign(), dt, spec.gain_scale, spec.ramp_fpm)
    ap.thr = ap.thr_i = fdm['fcs/throttle-cmd-norm[0]']   # start the autothrottle at the trimmed setting
    ac.set_wind(fdm, spec, spec.cruise_alt)
    for _ in range(int(5 * ac.DT_HZ)):              # settle the integrators, not logged
        u = ap.step(fdm, spec.cruise_alt, spec.start_cas, spec.heading)
        _apply(fdm, u, {f'thr{e}': u['throttle'] for e in range(ac.N_ENG)})
        fdm.run()
    ac.enable_turbulence(fdm, spec.seed, spec.turb_severity)
    S = Sensed(fdm, anom)
    names, nodes = property_nodes(fdm)

    n_steps = int(spec.duration_s * ac.DT_HZ)
    every = int(round(ac.DT_HZ / ac.LOG_HZ))
    n_log = (n_steps + every - 1) // every
    P = np.empty((n_log, len(nodes)), dtype=np.float64)
    extra = {k: np.empty(n_log) for k in
             ['t'] + [f'sensor/{v}' for v in SENSED] + [f'sensor/TLA{e + 1}' for e in range(ac.N_ENG)]
             + ['ap/elevator-cmd', 'ap/aileron-cmd', 'ap/rudder-cmd', 'ap/throttle-cmd', 'ap/target-alt-ft',
                'sel/alt-ft', 'sel/cas-kt', 'sel/hdg-deg', 'sel/mach']
             + [f'derived/engine[{e}]/egt-degc' for e in range(ac.N_ENG)]}
    j = 0
    for i in range(n_steps):
        t = i * dt
        S.step(t)
        sel_alt, sel_hdg = spec.sel_alt(t), spec.sel_hdg(t)
        sel_cas = spec.sel_cas(S['position/h-sl-ft'], t)
        u_cmd = dict(ap.step(S, sel_alt, sel_cas, sel_hdg))
        thr_cmd = {f'thr{e}': u_cmd['throttle'] for e in range(ac.N_ENG)}
        u_app = anom.actuate(t, {**{k: u_cmd[k] for k in CMD_PROPS}, **thr_cmd})
        _apply(fdm, u_app, u_app)
        if i % int(ac.DT_HZ) == 0:
            ac.set_wind(fdm, spec, fdm['position/h-sl-ft'])
        if not fdm.run():
            raise RuntimeError(f'JSBSim stopped at t={t:.1f}s')
        if i % every:
            continue
        S.step(t + 1e-9)                              # fresh readings after the step
        P[j] = [nd.get_double_value() for nd in nodes]
        x = extra
        x['t'][j] = t
        for v, (prop, _, _) in SENSED.items():
            val = S[prop]
            if prop.endswith('rad_sec'):
                val = np.degrees(val)
            elif v == 'VS':
                val *= 60.0
            elif v == 'HDG':
                val %= 360.0
            x[f'sensor/{v}'][j] = val
        for e in range(ac.N_ENG):
            x[f'sensor/TLA{e + 1}'][j] = anom.lever(e, thr_cmd[f'thr{e}'], u_app[f'thr{e}'])
            k = f'propulsion/engine[{e}]'
            x[f'derived/engine[{e}]/egt-degc'][j] = ac.egt_c(
                fdm['propulsion/tat-c'], fdm[f'fcs/throttle-pos-norm[{e}]'],
                fdm[f'{k}/set-running'] > 0.5, fdm[f'{k}/stalled'] > 0.5)
        x['ap/elevator-cmd'][j], x['ap/aileron-cmd'][j] = u_cmd['elevator'], u_cmd['aileron']
        x['ap/rudder-cmd'][j], x['ap/throttle-cmd'][j] = u_cmd['rudder'], u_cmd['throttle']
        x['ap/target-alt-ft'][j] = ap.tgt_alt
        x['sel/alt-ft'][j], x['sel/cas-kt'][j] = sel_alt, sel_cas
        x['sel/hdg-deg'][j], x['sel/mach'][j] = sel_hdg, spec.sel_mach(t)
        j += 1

    P = P[:j]
    P[~np.isfinite(P) | (np.abs(P) > 1e15)] = np.nan          # JSBSim 'unset' markers
    df = pd.concat([pd.DataFrame({k: v[:j] for k, v in extra.items()}),
                    pd.DataFrame(P, columns=names)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df.insert(1, 'label', label_of(anomaly, df['t'].to_numpy()))
    return df


def label_of(anomaly, t):
    """1 inside [onset, end); for an intermittent profile only during the bursts."""
    if not anomaly:
        return np.zeros(len(t), 'int8')
    lab = (t >= anomaly['onset_s']) & (t < anomaly['end_s'])
    if anomaly['profile'] == 'intermittent':
        lab &= envelope(anomaly, t) > 0
    return lab.astype('int8')


def _apply(fdm, u, thr):
    for k, p in CMD_PROPS.items():
        fdm[p] = float(min(max(u[k], -1.0), 1.0))
    for e in range(ac.N_ENG):
        fdm[f'fcs/throttle-cmd-norm[{e}]'] = float(min(max(thr[f'thr{e}'], 0.0), 1.0))


def add_sensor_noise(df, seed):
    """Measurement noise on the sensor/* channels only (same draw for a flight and
    its normal twin, so before onset the two are identical)."""
    rng = np.random.default_rng(seed + 11)
    df = df.copy()
    for c in [c for c in df.columns if c.startswith('sensor/')]:
        key = 'TLA' if c.startswith('sensor/TLA') else c.split('/')[1]
        df[c] = df[c] + rng.normal(0.0, SENSOR_NOISE[key], len(df))
    df['sensor/HDG'] = df['sensor/HDG'] % 360.0
    return df
