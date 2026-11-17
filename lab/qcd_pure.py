"""Build the QCD dataset the task-aware-compression spec asks for (task_aware_compression.pdf, slides 6-8):
pure normal trajectories D0, PURE ABNORMAL trajectories D1 (the anomaly is active from the first sample),
and change trajectories (normal, then the anomaly from a change point nu to the end) for test.

    python -m lab.qcd_pure --out C:/.../qcd_pure          # full set
    python -m lab.qcd_pure --out ... --scale 0.02         # smoke test
    python -m lab.qcd_pure --out ... --assemble-only

Design (differences from lab.qcd_dataset in CAPS)
* T = 1200 samples = 300 s at 4 Hz (spec), ONE trajectory per flight, the same window 720-1020 s in every
  flight. Flights are FLIGHT_S = 1100 s long (the window + envelope margin); no normal twin is flown.
* Roles (by whole flight, never by time step):
    train_normal 560 | val_normal 140 | test_normal 140
    train_abnormal 560 | val_abnormal 140 | test_abnormal 140   (40 / 10 / 10 per anomaly type, 14 types)
    val_change 140 | test_change 140                            (10 / 10 per type)
  ABNORMAL: onset = the first sample of the window (720 s), label = 1 for all T samples.
  CHANGE: onset at nu ~ U[NU_RANGE] samples inside the window (burn-in t0 + margin before, >= 400 samples
  = 100 s runway after), label = 1 from nu on. val_change is for measuring the post-change drift curve
  (spec slide 17, instead of spliced streams); test_change for detection rate and delay only.
* All 14 anomaly types with equal counts; variable, severity U(0,1), profile (step / ramp) and sign random;
  the anomaly lasts to the end of the flight. Hard conditions on 20 % of every role. Envelope rule and
  severity redraw exactly as lab.qcd_dataset.
* Features: every channel that changes in the training windows, MINUS clock/position bookkeeping and MINUS
  channels constant in normal flight (simulator artefacts such as the exactly-zero yawing moment of four
  identical engines, which would make raw detection trivial). Angles as sin/cos. No clipping, no dedup.
* Normalisation: z-score with train_normal statistics, std floored at 0.1 x all-train std (as before).
* Output is written as memory-mappable .npy (x float32 (N,T,D), label int8 (N,T)) plus <role>_meta.npz,
  streamed flight by flight so the build never holds a whole split in RAM.
"""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import anomalies as an
from .aircraft import LOG_HZ
from .qcd_dataset import (CIRCULAR, ENVELOPE_MARGIN_S, EXCLUDE_PREFIXES, HARD_FRAC, MAX_ATTEMPTS, RAMP_S,
                          SEV_SHRINK, choose_channels, envelope_exit, feature_names, featurise)
from .scenario import HARD_NORMALS, draw_spec
from .simulate import add_sensor_noise, fly

T_WIN = 1200
WIN_S = 720.0
FLIGHT_S = 1100.0
NU_RANGE = (100, 800)                      # change point (samples) inside the window: t0=40 + margin, runway 400
PROFILES = ('step', 'ramp')
SEED_BASE = 700_000
NORMAL = {'train': 560, 'val': 140, 'test': 140}
ABNORMAL_PER_TYPE = {'train': 40, 'val': 10, 'test': 10}
CHANGE_PER_TYPE = {'val': 10, 'test': 10}
ROLES = ('train_normal', 'val_normal', 'test_normal', 'train_abnormal', 'val_abnormal', 'test_abnormal',
         'val_change', 'test_change')


def plan(scale=1.0):
    def n(k):
        return max(1, int(round(k * scale)))
    tasks, i = [], 0
    for split, k in NORMAL.items():
        for j in range(n(k)):
            hard = HARD_NORMALS[j % len(HARD_NORMALS)] if j < round(n(k) * HARD_FRAC) else None
            tasks.append(dict(id=f'PN{i:04d}', seed=SEED_BASE + i, split=split, role=f'{split}_normal', hard=hard,
                              typ=None))
            i += 1
    for kind, per in (('abnormal', ABNORMAL_PER_TYPE), ('change', CHANGE_PER_TYPE)):
        n_hard = {}
        for typ in an.CATALOG:
            for split, k in per.items():
                role = f'{split}_{kind}'
                for j in range(n(k)):
                    hard = None
                    if j < round(n(k) * HARD_FRAC):
                        c = n_hard[role] = n_hard.get(role, -1) + 1
                        hard = HARD_NORMALS[c % len(HARD_NORMALS)]
                    tasks.append(dict(id=f'P{"A" if kind == "abnormal" else "C"}{i:04d}', seed=SEED_BASE + i,
                                      split=split, role=role, hard=hard, typ=typ))
                    i += 1
    return tasks


def change_point(task):
    if task['role'].endswith('_abnormal'):
        return 0
    if task['role'].endswith('_change'):
        return int(np.random.default_rng(task['seed'] + 13).integers(NU_RANGE[0], NU_RANGE[1] + 1))
    return -1


def draw_anomaly(task, attempt=0):
    rng = np.random.default_rng(task['seed'] + 7 + 1000 * attempt)
    typ = task['typ']
    c = an.CATALOG[typ]
    var = c['vars'][int(rng.integers(len(c['vars'])))]
    onset = WIN_S + change_point(task) / LOG_HZ
    prof = an._choice(rng, {p: an.PROFILE_W[p] for p in c['profiles'] if p in PROFILES})
    sev = float(rng.uniform()) * SEV_SHRINK ** attempt
    a = an.sample(rng, FLIGHT_S + an.TAIL_S, typ=typ, variable=None if var == 'engine' else var,
                  severity=sev, profile=prof, onset=onset, duration=FLIGHT_S - onset, cap=False)
    if a['profile'] == 'ramp':
        a['ramp_s'] = RAMP_S
    return a


def run(task, cache):
    path = cache / f"{task['id']}.npz"
    if path.exists():
        return task['id'], 'kept', None
    t0 = time.time()
    rejected = []
    s0 = int(round(WIN_S * LOG_HZ))
    need = int((WIN_S + T_WIN / LOG_HZ + ENVELOPE_MARGIN_S) * LOG_HZ)
    ok = False
    # if every severity redraw leaves the envelope (some faults, e.g. a pitot blockage in heavy turbulence,
    # do not scale with severity), re-seed the scenario (recorded as reseed) so the type counts stay equal
    for reseed in range(4 if task['typ'] else 1):
        seed = task['seed'] + 50_000 * reseed
        spec = draw_spec(task['id'], seed, task['hard'], duration=FLIGHT_S)
        for attempt in range(MAX_ATTEMPTS if task['typ'] else 1):
            a = draw_anomaly(dict(task, seed=seed), attempt) if task['typ'] else None
            df = add_sensor_noise(fly(spec, a), spec.seed)
            n, ex = len(df), envelope_exit(df)
            if n >= need and (ex is None or ex >= need):
                ok = True
                break
            rejected.append(dict(reseed=reseed, attempt=attempt,
                                 envelope_exit_s=round(ex / LOG_HZ, 2) if ex is not None else None, rows=n,
                                 anomaly={k: a.get(k) for k in ('type', 'variable', 'engine', 'severity',
                                                                'profile')} if a else None))
        if ok:
            break
    if not ok:
        raise RuntimeError(f"{task['id']}: left the envelope in all attempts and reseeds")
    task = dict(task, seed=seed, reseed=reseed)
    cols = [c for c in df.columns if c not in ('t', 'label')]
    w = df[cols].to_numpy(np.float32)[s0:s0 + T_WIN][None]
    wl = df['label'].to_numpy(np.int8)[s0:s0 + T_WIN][None]
    cp = change_point(task)
    if task['role'].endswith('_normal'):
        assert (wl == 0).all(), task['id']
    elif task['role'].endswith('_abnormal'):
        assert (wl == 1).all(), (task['id'], wl.mean())
    else:
        assert wl[0, :cp].max() == 0 and wl[0, cp:].min() == 1, task['id']
    meta = dict(task, anomaly=a, scenario=spec.summary(), n_rows=n,
                envelope_exit_s=round(ex / LOG_HZ, 2) if ex is not None else None, attempts=attempt + 1,
                rejected=rejected, wall_s=round(time.time() - t0, 1))
    np.savez_compressed(path, x=w, label=wl, start=np.array([s0]), cp=np.array([cp]), cols=np.array(cols),
                        meta=np.array(json.dumps(meta)))
    return task['id'], 'flown', round(time.time() - t0, 1)


def load_cache(cache, fid):
    z = np.load(cache / f'{fid}.npz')
    return z['x'], z['label'], z['cp'], list(z['cols']), json.loads(str(z['meta']))


def assemble(out, tasks, cache):
    # channels that change in the training windows (normal or abnormal), minus bookkeeping, minus NaN
    pseudo = [dict(t, split='train') if t['role'] in ('train_normal', 'train_abnormal') else dict(t, split='x')
              for t in tasks]
    cols, keep, dropped = _choose(cache, pseudo)
    circ = {c: u for c, u in CIRCULAR.items() if c in keep}
    idx = [(c, cols.index(c)) for c in keep]
    names = feature_names(keep, circ)
    by_role = {}
    for t in tasks:
        by_role.setdefault(t['role'], []).append(t)
    # streaming moments: train_normal (mean, std) and all-train std (normal + abnormal)
    def moments(roles):
        s1 = s2 = None
        cnt = 0
        for r in roles:
            for t in by_role[r]:
                x = featurise(load_cache(cache, t['id'])[0], idx, circ)[0].astype(np.float64)
                s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
                s2 = (x ** 2).sum(0) if s2 is None else s2 + (x ** 2).sum(0)
                cnt += len(x)
        m = s1 / cnt
        return m, np.sqrt(np.maximum(s2 / cnt - m ** 2, 0.0))
    mean, std_n = moments(['train_normal'])
    _, std_all = moments(['train_normal', 'train_abnormal'])
    const = std_n <= 1e-9 * np.maximum(1.0, np.abs(mean))
    dropped['constant in normal flight (simulator artefact)'] = [names[k] for k in np.where(const)[0]]
    sel = np.where(~const)[0]
    names, mean, std_n, std_all = [names[k] for k in sel], mean[sel], std_n[sel], std_all[sel]
    std = np.maximum.reduce([std_n, 0.1 * std_all, np.full_like(std_n, 1e-8)])
    floored = [names[k] for k in np.where(std_n < 0.1 * std_all)[0]]
    np.savez(out / 'normalization.npz', mean=mean.astype(np.float32), std=std.astype(np.float32),
             channels=np.array(names))
    counts = {}
    for role in ROLES:
        ts = by_role.get(role, [])
        if not ts:
            continue
        X = np.lib.format.open_memmap(out / f'{role}_x.npy', mode='w+', dtype=np.float32,
                                      shape=(len(ts), T_WIN, len(names)))
        L = np.zeros((len(ts), T_WIN), np.int8)
        info = {k: [] for k in ('flight_id', 'anomaly_type', 'group', 'variable', 'severity', 'profile', 'hard',
                                'change_point')}
        mx, big = 0.0, 0
        for i, t in enumerate(ts):
            x, lab, cp, _, meta = load_cache(cache, t['id'])
            z = ((featurise(x, idx, circ)[0][:, sel] - mean) / std).astype(np.float32)
            assert np.isfinite(z).all(), t['id']
            X[i] = z
            L[i] = lab[0]
            mx, big = max(mx, float(np.abs(z).max())), big + int((np.abs(z) > 10).sum())
            a = meta['anomaly'] or {}
            info['flight_id'].append(t['id'])
            info['anomaly_type'].append(a.get('type', 'normal'))
            info['group'].append(a.get('group', 'normal'))
            info['variable'].append(a.get('variable') or (f"engine {a['engine']}" if a.get('engine') else ''))
            info['severity'].append(a.get('severity', 0.0) or 0.0)
            info['profile'].append(a.get('profile', ''))
            info['hard'].append(t['hard'] or '')
            info['change_point'].append(int(cp[0]))
        X.flush()
        del X
        np.save(out / f'{role}_label.npy', L)
        np.savez(out / f'{role}_meta.npz', **{k: np.array(v) for k, v in info.items()})
        types = {}
        for ty in info['anomaly_type']:
            types[ty] = types.get(ty, 0) + 1
        counts[role] = dict(n=len(ts), shape=[len(ts), T_WIN, len(names)], label_fraction=float(L.mean()),
                            per_type=types, max_abs_z=mx, frac_abs_z_gt_10=big / (len(ts) * T_WIN * len(names)))
        print(f'{role:15s} ({len(ts)}, {T_WIN}, {len(names)})  label=1 share {L.mean():.3f}  max|z| {mx:.1f}',
              flush=True)
    walls = [load_cache(cache, t['id'])[-1].get('wall_s') for t in tasks]
    walls = [w for w in walls if w]
    meta = dict(created=time.strftime('%Y-%m-%d %H:%M'), source='cruise_lab (JSBSim 1.3.1, B747), lab.qcd_pure',
                T=T_WIN, rate_hz=LOG_HZ, window_s=[WIN_S, WIN_S + T_WIN / LOG_HZ], flight_s=FLIGHT_S,
                change_point_range=list(NU_RANGE), abnormal='anomaly active from sample 0 (onset = window start)',
                counts=counts, n_features=len(names), circular=sorted(circ),
                dropped={k: sorted(v) for k, v in dropped.items()}, normal_std_floored=floored, channels=names,
                normalisation='x = (raw - mean) / std, train_normal stats; std = max(std_n, 0.1 std_all_train, 1e-8)',
                wall_s_per_flight=dict(mean=float(np.mean(walls)), max=float(np.max(walls)), n=len(walls))
                if walls else None)
    (out / 'metadata.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    with open(out / 'flights.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['flight_id', 'seed', 'role', 'hard', 'anomaly_type', 'variable', 'engine', 'severity',
                    'profile', 'sign', 'onset_s', 'change_point', 'attempts', 'wall_s'])
        for t in tasks:
            _, _, cp, _, m = load_cache(cache, t['id'])
            a = m['anomaly'] or {}
            w.writerow([t['id'], t['seed'], t['role'], t['hard'] or '', a.get('type', ''), a.get('variable') or '',
                        a.get('engine') or '', a.get('severity', ''), a.get('profile', ''), a.get('sign', ''),
                        a.get('onset_s', ''), int(cp[0]), m.get('attempts', 1), m.get('wall_s', '')])
    print(f'{len(names)} features ({len(keep)} channels, {len(circ)} as sin/cos)')


def _choose(cache, tasks):
    """lab.qcd_dataset.choose_channels with bookkeeping dropped; it reads x/label/start/cp/cols/meta."""
    return choose_channels(cache, tasks, keep_bookkeeping=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--assemble-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    cache = out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    tasks = plan(args.scale)
    print(f'{len(tasks)} flights -> {out}', flush=True)
    if not args.assemble_only:
        t0, done = time.time(), 0
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(run, t, cache): t['id'] for t in tasks}
            for f in as_completed(futs):
                f.result()
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    print(f'  {done}/{len(tasks)} flights  ({time.time() - t0:.0f} s)', flush=True)
    assemble(out, tasks, cache)


if __name__ == '__main__':
    main()
