"""Generate cruise flights.

    python -m lab.generate                       # 'demo' preset: 4 normal + 8 observation + 8 control
    python -m lab.generate --preset catalog      # every anomaly type x variable + 10 normal
    python -m lab.generate --preset random --n 30
    python -m lab.generate --preset demo --out data --workers 6

Output (in --out, default ./data):
    flights/<id>.parquet        the flight: every logged channel at 4 Hz
    flights/<id>_twin.parquet   anomalous flights only: the SAME flight (same seed, same turbulence,
                                same sensor noise) flown without the anomaly, from 60 s before onset
    meta/<id>.json              scenario + anomaly + per-channel effect scores
    manifest.json               all flights (what the viewer lists)
    channels.json               every channel: subsystem, unit, meaning, whether it changes in flight

Existing flights are kept (re-running only adds what is missing).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import anomalies as an
from .scenario import HARD_NORMALS, draw_spec
from .simulate import add_sensor_noise, fly

HERE = Path(__file__).parent
DEMO = {
    'normal': [None, 'moderate_turbulence', 'strong_wind', 'heavy'],
    'observation': [('sensor_bias', 'ALT', 'step'), ('sensor_drift', 'CAS', 'ramp'),
                    ('sensor_stuck', 'VS', 'step'), ('sensor_noise', 'PITCH', 'step'),
                    ('sensor_spikes', 'HDG', 'step'), ('pitot_blockage', 'CAS', 'step'),
                    ('sensor_bias', 'BETA', 'intermittent'), ('sensor_drift', 'HDG', 'ramp')],
    'control': [('surface_bias', 'elevator', 'step'), ('surface_stuck', 'elevator', 'step'),
                ('surface_loss_of_effectiveness', 'elevator', 'step'), ('surface_oscillation', 'aileron', 'step'),
                ('surface_runaway', 'rudder', 'step'), ('throttle_thrust_loss', 'engine', 'intermittent'),
                ('throttle_lever_stuck', 'engine', 'step'), ('throttle_lever_bias', 'engine', 'step')],
}
SEVERITY = 0.7
ROLE_STRONG, ROLE_MODERATE, ROLE_WEAK = 3.0, 1.0, 0.25


# ------------------------------------------------------------------ plan

def plan(preset, n):
    tasks = []
    if preset in ('demo', 'catalog'):
        base = 100_000 if preset == 'demo' else 200_000
        normals = DEMO['normal'] if preset == 'demo' else [None, None, None, None, None] + list(HARD_NORMALS)
        for i, hard in enumerate(normals):
            tasks.append(dict(id=f'N{base // 100_000}{i:02d}', seed=base + i, hard=hard, anomaly=None))
        if preset == 'demo':
            lists = {g: DEMO[g] for g in ('observation', 'control')}
        else:
            lists = {g: [(t, v, c['profiles'][0]) for t, c in an.CATALOG.items() if c['group'] == g
                         for v in c['vars']] for g in ('observation', 'control')}
        for g, items in lists.items():
            pfx = 'O' if g == 'observation' else 'C'
            for i, (typ, var, prof) in enumerate(items):
                tasks.append(dict(id=f'{pfx}{base // 100_000}{i:02d}', seed=base + 1000 * (pfx == 'C') + 100 + i,
                                  hard=None, anomaly=dict(typ=typ, variable=None if var == 'engine' else var,
                                                          severity=SEVERITY, profile=prof)))
    elif preset == 'random':
        for i in range(n):
            g = ('normal', 'observation', 'control')[i % 3]
            pfx = {'normal': 'N', 'observation': 'O', 'control': 'C'}[g]
            tasks.append(dict(id=f'{pfx}3{i:03d}', seed=300_000 + i, hard=None,
                              anomaly=None if g == 'normal' else dict(group=g)))
    else:
        raise SystemExit(f'unknown preset {preset}')
    return tasks


# ------------------------------------------------------------------ one flight

def _store(df, path):
    keep64 = [c for c in df.columns if c == 't' or 'lat' in c or 'long' in c]
    out = df.astype({c: 'float32' for c in df.columns if c not in keep64 and c != 'label'})
    out.to_parquet(path, index=False, compression='zstd')


def effects(df, tw, a):
    """Per channel: how much the anomalous flight departs from its normal twin.
    score = max |flight - twin| after onset / std(twin channel over the flight);
    first_s = first time after onset the departure exceeds 0.5 std."""
    t = df['t'].to_numpy()
    tw = tw.set_index('t').reindex(df['t']).reset_index()
    post = t >= a['onset_s']
    pre = (t < a['onset_s']) & tw.iloc[:, 1].notna().to_numpy()
    out, pre_max = {}, 0.0
    for c in df.columns:
        if c in ('t', 'label'):
            continue
        x, y = df[c].to_numpy(float), tw[c].to_numpy(float)
        d = np.abs(x - y)
        if pre.any():
            pre_max = max(pre_max, float(np.nanmax(d[pre])) if np.isfinite(d[pre]).any() else 0.0)
        # spread of the channel in the normal twin, floored at 0.1 % of its level so
        # nearly-constant channels (wind, fuel density...) do not get huge scores, and at
        # 5 % of the flight's own spread for channels that are ~0 in normal flight
        scale = max(float(np.nanstd(y)), 1e-3 * abs(float(np.nanmean(y))), 0.05 * float(np.nanstd(x)), 1e-9)
        dp = d[post]
        if not np.isfinite(dp).any():
            continue
        score = min(float(np.nanmax(dp)) / scale, 999.0)
        if score < 0.05:
            continue
        hit = np.where(dp > 0.5 * scale)[0]
        first = round(float(t[post][hit[0]] - a['onset_s']), 2) if len(hit) else None
        out[c] = [round(score, 3), first]
    return out, pre_max


def role(score, injected):
    if injected:
        return 'injection point'
    if score >= ROLE_STRONG:
        return 'strong response'
    if score >= ROLE_MODERATE:
        return 'moderate response'
    if score >= ROLE_WEAK:
        return 'weak response'
    return 'no visible effect'


def run(task, out):
    fp = out / 'flights' / f"{task['id']}.parquet"
    mp = out / 'meta' / f"{task['id']}.json"
    if fp.exists() and mp.exists():
        return task['id'], 'kept'
    t0 = time.time()
    spec = draw_spec(task['id'], task['seed'], task['hard'])
    a = None
    if task['anomaly']:
        rng = np.random.default_rng(task['seed'] + 7)
        req = task['anomaly']
        a = an.sample(rng, spec.duration_s, group=req.get('group'), typ=req.get('typ'),
                      variable=req.get('variable'), severity=req.get('severity'), profile=req.get('profile'))
    df = add_sensor_noise(fly(spec, a), spec.seed)
    meta = dict(id=task['id'], seed=task['seed'], kind='normal' if a is None else a['group'],
                hard_normal=spec.hard_normal, scenario=spec.summary(), anomaly=a,
                rows=len(df), duration_s=float(df['t'].iloc[-1]),
                injection_channels=an.injection_channels(a))
    ranges = (df.max(numeric_only=True) - df.min(numeric_only=True)).abs()
    meta['changing'] = sorted(c for c, r in ranges.items() if np.isfinite(r) and r > 1e-9 and c not in ('t',))
    if a is not None:
        tw = add_sensor_noise(fly(spec, None), spec.seed)
        eff, pre_max = effects(df, tw, a)
        meta['twin_identical_before_onset'] = bool(pre_max == 0.0)
        meta['effects'] = {c: dict(score=s, first_s=f, role=role(s, c in meta['injection_channels']))
                           for c, (s, f) in eff.items()}
        for c in meta['injection_channels']:
            meta['effects'].setdefault(c, dict(score=None, first_s=0.0, role='injection point'))
        _store(tw[tw['t'] >= a['onset_s'] - 60.0], out / 'flights' / f"{task['id']}_twin.parquet")
    fp.parent.mkdir(parents=True, exist_ok=True)
    _store(df, fp)
    meta['wall_s'] = round(time.time() - t0, 1)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(meta, indent=1), encoding='utf-8')
    return task['id'], 'ok'


# ------------------------------------------------------------------ manifest + channel table

UNITS = [('-rad_sec2', 'rad/s²'), ('-rad_sec', 'rad/s'), ('-deg_sec', 'deg/s'), ('-ft_sec2', 'ft/s²'),
         ('-fps2', 'ft/s²'), ('-fps', 'ft/s'), ('-kts', 'kt'), ('-kt', 'kt'), ('-ft', 'ft'), ('-deg', 'deg'),
         ('-rad', 'rad'), ('-lbs_per_gal', 'lb/gal'), ('-lbs', 'lb'), ('-psf', 'lb/ft²'), ('-inhg', 'inHg'),
         ('-slug_ft2', 'slug·ft²'), ('-slugs_ft3', 'slug/ft³'), ('-slugs', 'slug'), ('-gal', 'gal'),
         ('-pps', 'lb/s'), ('-gph', 'gal/h'), ('-norm', 'normalised'), ('-degc', '°C'), ('-c', '°C'),
         ('-R', '°R'), ('_ft', 'ft'), ('-sec', 's'), ('-mph', 'mph')]
SENSOR_UNITS = {'ALT': 'ft', 'CAS': 'kt', 'VS': 'ft/min', 'PITCH': 'deg', 'ROLL': 'deg', 'HDG': 'deg',
                'BETA': 'deg', 'Q': 'deg/s', 'P': 'deg/s'}
SPECIAL = {
    'sensor/ALT': 'Altimeter reading fed to the autopilot (anomaly + noise)',
    'sensor/CAS': 'Airspeed reading (calibrated) fed to the autothrottle (anomaly + noise)',
    'sensor/VS': 'Vertical-speed reading fed to the pitch loop (anomaly + noise)',
    'sensor/PITCH': 'Pitch attitude reading (anomaly + noise)',
    'sensor/ROLL': 'Roll attitude reading (anomaly + noise)',
    'sensor/HDG': 'Heading reading fed to the heading loop (anomaly + noise)',
    'sensor/BETA': 'Sideslip reading fed to the rudder loop (anomaly + noise)',
    'sensor/Q': 'Pitch-rate gyro reading (anomaly + noise)',
    'sensor/P': 'Roll-rate gyro reading (anomaly + noise)',
    'ap/elevator-cmd': 'Autopilot elevator command BEFORE any control anomaly (-1..1)',
    'ap/aileron-cmd': 'Autopilot aileron command BEFORE any control anomaly (-1..1)',
    'ap/rudder-cmd': 'Autopilot rudder command (sideslip loop) BEFORE any control anomaly (-1..1)',
    'ap/throttle-cmd': 'Autothrottle command, same for all engines, BEFORE any control anomaly (0..1)',
    'ap/target-alt-ft': 'Autopilot internal altitude target (selected altitude, rate-limited)',
    'sel/alt-ft': 'Selected altitude (flight plan)', 'sel/cas-kt': 'Selected CAS (from selected Mach)',
    'sel/hdg-deg': 'Selected heading (flight plan, route turns)', 'sel/mach': 'Selected Mach',
    **{f'sensor/TLA{e}': f'Thrust-lever angle of engine {e} as recorded (0..1); frozen if that lever is stuck, '
                         'normal for thrust-loss / lever-bias faults' for e in range(1, 5)},
    **{f'derived/engine[{e}]/egt-degc': f'Exhaust gas temperature, engine {e + 1}: JSBSim turbine law '
                                        'TAT + 363.1 + 357.1 x throttle position' for e in range(4)},
}


def subsystem(c):
    import re
    if c.startswith('sensor/TLA'):
        return f'Engine {c[-1]}'
    if c.startswith('sensor/'):
        return 'Sensors (autopilot inputs)'
    if c.startswith('ap/'):
        return 'Autopilot commands'
    if c.startswith('sel/'):
        return 'Flight plan (selected)'
    m = re.search(r'engine\[(\d)\]', c) or re.match(r'fcs/(?:throttle|mixture|advance|feather)-\w+-norm\[(\d)\]', c)
    if m:
        return f'Engine {int(m.group(1)) + 1}'
    if c.startswith('propulsion/tank'):
        return 'Fuel tanks'
    ns = c.split('/')[0]
    return {'position': 'Position', 'velocities': 'Velocities & rates', 'attitude': 'Attitude',
            'accelerations': 'Accelerations', 'aero': 'Aerodynamics', 'fcs': 'Flight controls (surfaces)',
            'forces': 'Forces', 'moments': 'Moments', 'atmosphere': 'Atmosphere & wind', 'inertia': 'Mass & inertia',
            'flight-path': 'Flight path', 'propulsion': 'Propulsion (total)', 'metrics': 'Geometry constants',
            'systems': 'Aircraft systems', 'simulation': 'Simulator', 'ic': 'Initial conditions',
            'inertial': 'Earth model', 'orbital': 'Earth model', 'derived': 'Derived'}.get(ns, ns)


# subsystems that describe the environment or simulator bookkeeping rather than the
# aircraft's own systems: they 'respond' only because the aircraft is somewhere else
CONTEXT = ('Atmosphere & wind', 'Initial conditions', 'Geometry constants', 'Simulator', 'Earth model',
           'Position', 'Mass & inertia', 'Fuel tanks')


def unit_of(c):
    if c.startswith('sensor/'):
        return 'normalised' if 'TLA' in c else SENSOR_UNITS[c.split('/')[1]]
    if c.endswith('egt-degc'):
        return '°C'
    base = c.split('[')[0] if c.endswith(']') else c
    for suf, u in UNITS:
        if base.endswith(suf):
            return u
    if 'mach' in c:
        return 'Mach'
    return ''


def write_index(out):
    metas = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'meta').glob('*.json'))]
    order = {'normal': 0, 'observation': 1, 'control': 2}
    metas.sort(key=lambda m: (order[m['kind']], m['id']))
    man = [{k: m[k] for k in ('id', 'seed', 'kind', 'hard_normal', 'anomaly', 'duration_s', 'rows',
                              'injection_channels')} for m in metas]
    (out / 'manifest.json').write_text(json.dumps(man, indent=1), encoding='utf-8')

    inv = pd.read_csv(HERE / 'property_info.csv')
    from .simulate import canonical_names
    inv['name'] = canonical_names(inv['property'].tolist())
    info = inv.set_index('name')
    cols = pd.read_parquet(out / 'flights' / f"{metas[0]['id']}.parquet").columns
    count = {}
    for m in metas:
        for c in m['changing']:
            count[c] = count.get(c, 0) + 1
    rows = []
    for c in cols:
        if c in ('t', 'label'):
            continue
        r = info.loc[c] if c in info.index else None
        status = 'logged by cruise_lab' if r is None or not isinstance(r['status'], str) else r['status']
        rows.append(dict(name=c, subsystem=subsystem(c), unit=unit_of(c), context=subsystem(c) in CONTEXT,
                         meaning=SPECIAL.get(c) or ('' if r is None or not isinstance(r['meaning'], str)
                                                    else r['meaning']),
                         status=status, changing=c in count, n_flights_changing=count.get(c, 0)))
    (out / 'channels.json').write_text(json.dumps(rows, indent=0), encoding='utf-8')
    pd.DataFrame(rows).to_csv(out / 'channels.csv', index=False)
    write_channel_doc(rows, len(metas))
    return len(metas), sum(r['changing'] for r in rows), len(rows)


def write_channel_doc(rows, n_flights):
    """docs/03_channels.md: every channel that changes in flight, by subsystem."""
    df = pd.DataFrame(rows)
    chg = df[df.changing]
    order = ['Sensors (autopilot inputs)', 'Autopilot commands', 'Flight plan (selected)', 'Flight controls (surfaces)',
             'Engine 1', 'Engine 2', 'Engine 3', 'Engine 4', 'Attitude', 'Velocities & rates', 'Position',
             'Aerodynamics', 'Accelerations', 'Forces', 'Moments', 'Propulsion (total)', 'Flight path', 'Derived']
    subs = order + sorted(set(chg.subsystem) - set(order))
    L = ['# 3. Logged channels', '',
         f'_Generated by `lab/generate.py` from the current data ({n_flights} flights). Do not edit by hand._', '',
         f'Every flight file holds **{len(df)} channels** (plus `t` and `label`). **{len(chg)}** of them change '
         f'during at least one flight; the other {len(df) - len(chg)} are constants (geometry, limits, settings) '
         'and are kept only for completeness. The viewer lists the changing ones; tick *list constant channels too* '
         'to see all.', '',
         'Channel name prefixes:', '',
         '| Prefix | What it is |', '|---|---|',
         '| `sensor/` | What the sensors READ, i.e. what the autopilot uses (and an FDR records): true value + observation anomaly + noise |',
         '| `ap/` | Autopilot commands BEFORE any control anomaly, and its internal altitude target |',
         '| `sel/` | Flight plan: selected altitude, CAS, heading, Mach |',
         '| `derived/` | Computed from JSBSim internals (EGT law, not exposed as a property) |',
         '| anything else | A JSBSim property: the TRUE simulated state (`[0]` is written explicitly for the first engine / tank) |',
         '', '`context` = environment or bookkeeping channel: it can change after an anomaly only because the aircraft is '
         'somewhere else (different altitude -> different wind/air density). The viewer leaves these out of the '
         '"most affected" ranking by default.', '',
         '## Count by subsystem', '', '| Subsystem | Changing | Context |', '|---|---|---|']
    for s in subs:
        g = chg[chg.subsystem == s]
        if len(g):
            L.append(f'| {s} | {len(g)} | {"yes" if g.context.iloc[0] else ""} |')
    for s in subs:
        g = chg[chg.subsystem == s]
        if not len(g):
            continue
        L += ['', f'## {s}', '', '| Channel | Unit | Meaning / status | Changes in |', '|---|---|---|---|']
        for r in g.itertuples():
            m = r.meaning if r.meaning else ''
            st = '' if r.status in ('logged by cruise_lab', 'constant in flight') else r.status
            L.append(f'| `{r.name}` | {r.unit} | {m}{" · " + st if st and st not in m else ""} | {r.n_flights_changing}/{n_flights} |')
    (HERE.parent / 'docs').mkdir(exist_ok=True)
    (HERE.parent / 'docs' / '03_channels.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', default='demo', choices=('demo', 'catalog', 'random'))
    ap.add_argument('--n', type=int, default=30, help='number of flights for --preset random')
    ap.add_argument('--out', default=str(HERE.parent / 'data'))
    ap.add_argument('--workers', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()
    out = Path(args.out)
    tasks = plan(args.preset, args.n)
    print(f'{len(tasks)} flights -> {out}  ({args.workers} workers)', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, t, out): t['id'] for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                fid, st = f.result()
            except Exception as e:                    # keep going; report at the end
                fid, st = futs[f], f'ERROR {e!r}'
            print(f'[{i}/{len(tasks)}] {fid} {st}  ({time.time() - t0:.0f} s)', flush=True)
    n, n_chg, n_all = write_index(out)
    print(f'done: {n} flights in manifest; {n_chg} of {n_all} channels change in flight')


if __name__ == '__main__':
    main()
