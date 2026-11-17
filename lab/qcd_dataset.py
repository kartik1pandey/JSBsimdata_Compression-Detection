"""Build a training dataset for the task-aware QCD code (supervisor's
`task(QCD)_aware_data_compression` project) from Cruise Lab flights.

    python -m lab.qcd_dataset                     # full set, 980 flights
    python -m lab.qcd_dataset --scale 0.1         # quick smoke test
    python -m lab.qcd_dataset --assemble-only     # redo stage 2 from the cache

Output (default ./qcd_data), the same layout as the simulated VAR data that
data.py loads, plus extra keys the loader ignores:

    train_normal.npz  val_normal.npz  test_normal.npz       x (N, T, D), normal throughout
    train_change.npz  val_change.npz  test_change.npz       x (N, T, D) + change_point (N,): normal, then
                                                            the anomaly starts inside the sample
    every file also: label (N, T) int8, flight_id, anomaly_type, group, variable,
                     severity, profile, window_start_s (N,)
    normalization.npz   mean, std, channels   (x = (raw - mean) / std)
    flights.csv         one row per flight: split, role, anomaly, seed
    metadata.json       config, channel list with units, counts, exclusions

Design
------
* Windows of T = 480 samples (120 s at 4 Hz). Every flight is FLIGHT_S =
  1500 s long and every flight gives ONE sample, always 720-840 s.
  840/210/210 normal and 840/210/210 change flights (train/val/test).
  There are no fully-abnormal samples: every anomalous sample is a change
  sample (normal first, anomaly from the change point to the end).
* All 14 anomaly types, equal number of flights and windows per type;
  variable, engine, severity U(0, 1), profile (step or ramp) and sign drawn at
  random. Abnormal flights: onset U(400, 700) s, so the anomaly acts during
  every window. Change flights: onset at 760-800 s, inside their window. The
  anomaly lasts until the end of the flight (no intermittent gaps).
* Hard conditions (turbulence, wind, hot, cold, heavy) are given to 20 % of
  the flights in every role, normal and abnormal alike, so weather alone
  says nothing about the label.
* Flight envelope: the aircraft must stay inside it until the last window
  ends + 10 s; otherwise the anomaly is redrawn with severity x 0.7 per try.
* Every flight has exactly one role; no flight is shared between splits.
* Channels: every channel that changes in the training windows (nothing else
  removed by default; --drop-bookkeeping, --drop-const, --dedup are
  optional). Heading-like angles are replaced by sin/cos pairs so
  359 -> 0 deg is not a jump.
* Normalisation: z-score with train_normal statistics; std floored at
  0.1 x the all-train std for channels that are ~constant in normal flight.
  No clipping unless --clip.

Stage 1 (parallel) flies each flight once and caches its windows for all
logged channels in cache/<id>.npz; stage 2 selects channels, normalises and
writes the splits.
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
from .scenario import HARD_NORMALS, draw_spec
from .simulate import add_sensor_noise, fly

HERE = Path(__file__).parent
T_WIN = 480                                # samples per window (120 s at 4 Hz)
CP_RANGE = (T_WIN // 3, 2 * T_WIN // 3)    # change point position inside a change window
FLIGHT_S = 1500.0                          # every flight, normal or abnormal
ONSET_RANGE = (400.0, 700.0)               # abnormal onset (and normal window-region start)
FLIGHTS = {                                # flights per role; normal = 14 types x abnormal per type
    'normal': {'train': 840, 'val': 210, 'test': 210},
    'change_per_type': {'train': 60, 'val': 15, 'test': 15},     # change flights only (no fully-abnormal samples)
}
WIN_PER_FLIGHT = 1                         # one sample per flight, every flight alike
WIN_STARTS_S = (720.0,)                    # the SAME window (720-840 s) in every flight
CHANGE_WIN_S = 720.0                       # the one window of every change flight (same time)
PROFILES = ('step', 'ramp')                # anomaly acts continuously to the end
RAMP_S = 60.0                              # a ramp reaches full size 60 s after onset (inside the sample)
SEV_SHRINK = 0.7                           # severity factor per redraw after an envelope exit
CLIP_Z = 10.0                              # with --clip: standardised values clipped to +-CLIP_Z
DUP_CORR = 0.9999                          # |corr| above this on ALL train data = same signal
HARD_FRAC = 0.2
SEED_BASE = 500_000
ROLES_ORDER = ('train_normal', 'val_normal', 'test_normal', 'train_change', 'val_change', 'test_change')

# bookkeeping that encodes clock time or absolute place, not aircraft behaviour
EXCLUDE_PREFIXES = ('simulation/', 'orbital/', 'position/lat', 'position/long', 'position/eci-',
                    'position/ecef-', 'position/distance-from-start', 'position/from-start',
                    'position/vrp-', 'position/epa', 'position/radius-to-vehicle',
                    'position/terrain-elevation', 'inertial/')
# angles that wrap at 360 deg / 2 pi (heading, track, wind direction)
CIRCULAR = {'sensor/HDG': 'deg', 'sel/hdg-deg': 'deg', 'attitude/psi-deg': 'deg',
            'attitude/psi-rad': 'rad', 'attitude/heading-true-rad': 'rad',
            'flight-path/psi-gt-rad': 'rad', 'atmosphere/psiw-rad': 'rad'}


# ------------------------------------------------------------------ plan

def plan(scale=1.0):
    def n(k):
        return max(1, int(round(k * scale)))
    tasks, i = [], 0
    for split, k in FLIGHTS['normal'].items():
        for j in range(n(k)):
            hard = HARD_NORMALS[j % len(HARD_NORMALS)] if j < round(n(k) * HARD_FRAC) else None
            tasks.append(dict(id=f'QN{i:04d}', seed=SEED_BASE + i, split=split, role=f'{split}_normal',
                              hard=hard, typ=None))
            i += 1
    n_hard = {}                                      # rotate conditions across the whole role, not per type
    for typ in an.CATALOG:
        for role, k in FLIGHTS['change_per_type'].items():
            split = role
            for j in range(n(k)):
                hard = None
                if j < round(n(k) * HARD_FRAC):
                    c = n_hard[role] = n_hard.get(role, -1) + 1
                    hard = HARD_NORMALS[c % len(HARD_NORMALS)]
                tasks.append(dict(id=f'QA{i:04d}', seed=SEED_BASE + i, split=split,
                                  role=f'{split}_change',
                                  hard=hard, typ=typ))
                i += 1
    return tasks


def change_point(task):
    """Change point (sample inside the change window) of a change flight."""
    return int(np.random.default_rng(task['seed'] + 13).integers(CP_RANGE[0], CP_RANGE[1] + 1))


def draw_anomaly(task, attempt=0):
    rng = np.random.default_rng(task['seed'] + 7 + 1000 * attempt)
    typ = task['typ']
    c = an.CATALOG[typ]
    var = c['vars'][int(rng.integers(len(c['vars'])))]
    onset = float(rng.uniform(*ONSET_RANGE))
    if task['role'].endswith('_change'):             # onset inside the fixed change window
        onset = CHANGE_WIN_S + change_point(task) / LOG_HZ
    prof = an._choice(rng, {p: an.PROFILE_W[p] for p in c['profiles'] if p in PROFILES})
    sev = float(rng.uniform()) * SEV_SHRINK ** attempt
    a = an.sample(rng, FLIGHT_S + an.TAIL_S, typ=typ, variable=None if var == 'engine' else var,
                  severity=sev, profile=prof, onset=onset, duration=FLIGHT_S - onset, cap=False)
    if a['profile'] == 'ramp':
        a['ramp_s'] = RAMP_S
    return a


# ------------------------------------------------------------------ stage 1: fly one flight

# flight envelope: beyond this a crew would have taken over, so the data stops there
ENVELOPE = dict(bank_deg=45.0, pitch_deg=(-15.0, 25.0), alt_err_ft=3000.0, mach_max=0.92, cas_min_kt=160.0)
ENVELOPE_MARGIN_S = 10.0
MAX_ATTEMPTS = 8


def envelope_exit(df):
    """Index of the first row outside ENVELOPE (or with NaN in a core channel), else None."""
    e = ENVELOPE
    phi, th = df['attitude/phi-deg'].to_numpy(), df['attitude/theta-deg'].to_numpy()
    err = (df['position/h-sl-ft'] - df['sel/alt-ft']).to_numpy()
    mach, cas = df['velocities/mach'].to_numpy(), df['velocities/vc-kts'].to_numpy()
    with np.errstate(invalid='ignore'):
        bad = ((np.abs(phi) > e['bank_deg']) | (th < e['pitch_deg'][0]) | (th > e['pitch_deg'][1])
               | (np.abs(err) > e['alt_err_ft']) | (mach > e['mach_max']) | (cas < e['cas_min_kt'])
               | ~np.isfinite(phi + th + err + mach + cas))
    hit = np.where(bad)[0]
    return int(hit[0]) if len(hit) else None


def windows(task):
    """Fixed window starts (samples) and change points: identical for every flight of a kind."""
    if task['role'].endswith('_change'):
        return [int(round(CHANGE_WIN_S * LOG_HZ))], [change_point(task)]
    return [int(round(w * LOG_HZ)) for w in WIN_STARTS_S], [-1] * WIN_PER_FLIGHT


def run(task, cache):
    path = cache / f"{task['id']}.npz"
    if path.exists():
        return task['id'], 'kept', None
    t0 = time.time()
    spec = draw_spec(task['id'], task['seed'], task['hard'], duration=FLIGHT_S)
    rejected = []
    for attempt in range(MAX_ATTEMPTS if task['typ'] else 1):
        a = draw_anomaly(task, attempt) if task['typ'] else None
        df = add_sensor_noise(fly(spec, a), spec.seed)
        n, ex = len(df), envelope_exit(df)
        cut_s = round(ex / LOG_HZ, 2) if ex is not None else None
        last = CHANGE_WIN_S if task['role'].endswith('_change') else max(WIN_STARTS_S)
        need = int((last + T_WIN / LOG_HZ + ENVELOPE_MARGIN_S) * LOG_HZ)   # end of the data used
        if n >= int(FLIGHT_S * LOG_HZ) and (ex is None or ex >= need):   # complete flight; in envelope
            break                                                         # until the last window ends
        rejected.append(dict(attempt=attempt, envelope_exit_s=cut_s, rows=n,
                             anomaly={k: a.get(k) for k in ('type', 'variable', 'engine', 'severity', 'profile')}
                             if a else None))
    else:
        raise RuntimeError(f"{task['id']}: left the envelope in all {MAX_ATTEMPTS if task['typ'] else 1} attempts")
    starts, cps = windows(task)
    cols = [c for c in df.columns if c not in ('t', 'label')]
    X = df[cols].to_numpy(np.float32)
    lab = df['label'].to_numpy(np.int8)
    w = np.stack([X[s:s + T_WIN] for s in starts])
    wl = np.stack([lab[s:s + T_WIN] for s in starts])
    assert np.isfinite(w).sum() > 0.99 * w.size, task['id']
    if task['role'].endswith('_change'):             # label switches exactly at the change point
        assert wl[0, :cps[0]].max() == 0 and wl[0, cps[0]:].min() == 1, task['id']
    else:
        assert (wl == (1 if task['typ'] else 0)).all(), task['id']
    meta = dict(task, anomaly=a, scenario=spec.summary(), n_rows=n, envelope_exit_s=cut_s,
                attempts=attempt + 1, rejected=rejected)
    np.savez_compressed(path, x=w, label=wl, start=np.array(starts), cp=np.array(cps),
                        cols=np.array(cols), meta=np.array(json.dumps(meta)))
    return task['id'], 'flown', round(time.time() - t0, 1)


# ------------------------------------------------------------------ stage 2: assemble

def load_cache(cache, fid):
    z = np.load(cache / f'{fid}.npz')
    return z['x'], z['label'], z['start'], z['cp'], list(z['cols']), json.loads(str(z['meta']))


def choose_channels(cache, tasks, keep_bookkeeping=True):
    """Channels that change inside the training windows (normal or abnormal),
    minus EXCLUDE_PREFIXES, minus any channel with NaN."""
    lo = hi = None
    nan = None
    cols = None
    for t in tasks:
        if t['split'] != 'train':
            continue
        x, _, _, _, c, _ = load_cache(cache, t['id'])
        cols = cols or c
        assert c == cols, 'channel order differs between flights'
        bad = np.isnan(x).any(axis=(0, 1))
        nan = bad if nan is None else nan | bad
        with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore', RuntimeWarning)
            spread = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)      # spread inside each window
        spread = np.nan_to_num(spread, nan=0.0).max(axis=0)
        lo = spread if lo is None else np.maximum(lo, spread)
    changing = lo > 1e-6 * np.maximum(1.0, np.abs(lo))   # below this: float round-off, not a change
    keep, dropped = [], {'not changing': [], 'excluded bookkeeping': [], 'contains NaN': []}
    for k, c in enumerate(cols):
        if not changing[k]:
            dropped['not changing'].append(c)
        elif c.startswith(EXCLUDE_PREFIXES) and not keep_bookkeeping:
            dropped['excluded bookkeeping'].append(c)
        elif nan[k]:
            dropped['contains NaN'].append(c)
        else:
            keep.append(c)
    return cols, keep, dropped


def featurise(x, idx, circ):
    """Select channels; replace circular ones by sin/cos (in radians)."""
    out = []
    for name, k in idx:
        v = x[..., k]
        if name in circ:
            r = np.radians(v) if circ[name] == 'deg' else v
            out += [np.sin(r), np.cos(r)]
        else:
            out.append(v)
    return np.stack(out, axis=-1).astype(np.float32)


def feature_names(keep, circ):
    names = []
    for c in keep:
        names += [f'{c}:sin', f'{c}:cos'] if c in circ else [c]
    return names


def assemble(out, tasks, cache=None, noise_z=0.0, clip=False, dedup=False, drop_const=False,
             keep_bookkeeping=True):
    """noise_z > 0 adds white measurement noise of that std (in standardised units) to every
    channel that has none yet (all but sensor/*), so no channel is predictable to arbitrary
    precision; real recorders quantise and add noise to everything."""
    cache = Path(cache) if cache else out / 'cache'
    cols, keep, dropped = choose_channels(cache, tasks, keep_bookkeeping)
    circ = {c: u for c, u in CIRCULAR.items() if c in keep}
    idx = [(c, cols.index(c)) for c in keep]
    names = feature_names(keep, circ)
    by_role = {}
    for t in tasks:
        by_role.setdefault(t['role'], []).append(t)

    def gather(role):
        xs, ls, info = [], [], {k: [] for k in ('flight_id', 'anomaly_type', 'group', 'variable', 'severity',
                                                 'profile', 'window_start_s', 'change_point')}
        for t in by_role.get(role, []):
            x, lab, starts, cps, _, meta = load_cache(cache, t['id'])
            xs.append(featurise(x, idx, circ))
            ls.append(lab)
            a = meta['anomaly'] or {}
            for s, cp in zip(starts, cps):
                info['flight_id'].append(t['id'])
                info['anomaly_type'].append(a.get('type', 'normal'))
                info['group'].append(a.get('group', 'normal'))
                var = a.get('variable') or (f"engine {a['engine']}" if a.get('engine') else '')
                info['variable'].append(var)
                info['severity'].append(a.get('severity', 0.0) or 0.0)
                info['profile'].append(a.get('profile', ''))
                info['window_start_s'].append(s / LOG_HZ)
                info['change_point'].append(cp)
        return np.concatenate(xs), np.concatenate(ls), info

    # normalisation from train_normal (and the all-train spread for the floor)
    # one split in memory at a time; all-train std from pooled moments (no concatenation)
    xn, _, _ = gather('train_normal')
    sub = [xn[:, ::8]]                               # time-subsample kept for the duplicate test
    mean = xn.mean(axis=(0, 1), dtype=np.float64)
    std_n = xn.std(axis=(0, 1), dtype=np.float64)
    cnt = xn.shape[0] * xn.shape[1]
    s1, s2 = mean * cnt, (std_n ** 2 + mean ** 2) * cnt
    del xn
    xa, _, _ = gather('train_change')
    sub.append(xa[:, ::8])
    ma, va = xa.mean(axis=(0, 1), dtype=np.float64), xa.var(axis=(0, 1), dtype=np.float64)
    ca = xa.shape[0] * xa.shape[1]
    del xa
    cnt, s1, s2 = cnt + ca, s1 + ma * ca, s2 + (va + ma ** 2) * ca
    std_all = np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0))
    # a channel that never moves in normal flight but moves under a fault is a perfect
    # indicator no real recorder would give (e.g. the propulsive yawing moment is exactly
    # 0 because the four simulated engines are identical): drop it
    const = std_n <= 1e-9 * np.maximum(1.0, np.abs(mean))
    const_in_normal = [names[k] for k in np.where(const)[0]]
    if drop_const:
        dropped['constant in normal flight (simulator artefact)'] = const_in_normal
    else:
        const[:] = False
    sel = np.where(~const)[0]
    # the same signal in two units (alphadot rad/s and deg/s, psi deg and rad, ...) counts twice
    # in a per-channel likelihood; keep the first of each group. Tested on normal AND abnormal
    # train data, so a sensor reading is never merged with the truth it stops following in a fault.
    keep_mask, dup_of = np.ones(len(sel), bool), {}
    if dedup:
        v = np.concatenate(sub).reshape(-1, len(names))[:, sel].astype(np.float64)
        v = (v - v.mean(0)) / np.maximum(v.std(0), 1e-12)
        corr = np.abs(v.T @ v) / len(v)
        del v
    del sub
    for i in range(len(sel) if dedup else 0):
        if keep_mask[i]:
            for j in np.where(corr[i, i + 1:] > DUP_CORR)[0] + i + 1:
                if keep_mask[j]:
                    keep_mask[j] = False
                    dup_of[names[sel[j]]] = names[sel[i]]
    if dedup:
        dropped['duplicate of another channel (|corr| > %g)' % DUP_CORR] = [f'{a} = {b}' for a, b in dup_of.items()]
    sel = sel[keep_mask]
    names, mean, std_n, std_all = [names[k] for k in sel], mean[sel], std_n[sel], std_all[sel]
    std = np.maximum.reduce([std_n, 0.1 * std_all, np.full_like(std_n, 1e-8)])
    floored = [names[k] for k in np.where(std_n < 0.1 * std_all)[0]]
    np.savez(out / 'normalization.npz', mean=mean.astype(np.float32), std=std.astype(np.float32),
             channels=np.array(names))

    counts, ch_stats = {}, {}
    for role in ROLES_ORDER:
        if role not in by_role:
            continue
        x, lab, info = gather(role)
        x = ((x[..., sel] - mean) / std).astype(np.float32)
        if noise_z > 0:
            quiet = np.array([not n.startswith('sensor/') for n in names])
            rng = np.random.default_rng(SEED_BASE + 17 + ROLES_ORDER.index(role))
            x[..., quiet] += rng.normal(0.0, noise_z, x[..., quiet].shape).astype(np.float32)
        clipped_frac = float((np.abs(x) > CLIP_Z).mean())
        if clip:
            np.clip(x, -CLIP_Z, CLIP_Z, out=x)
        ch_stats[role] = dict(min=x.min(axis=(0, 1)), max=x.max(axis=(0, 1)))
        assert np.isfinite(x).all(), role
        arrays = dict(x=x, label=lab, **{k: np.array(v) for k, v in info.items()})
        if not role.endswith('_change'):
            arrays.pop('change_point')
        np.savez(out / f'{role}.npz', **arrays)
        types = {}
        for ty in info['anomaly_type']:
            types[ty] = types.get(ty, 0) + 1
        counts[role] = dict(windows=int(len(x)), flights=len(by_role[role]), shape=list(x.shape),
                            label_fraction=round(float(lab.mean()), 4), per_type=types,
                            fraction_beyond_10=round(clipped_frac, 6), max_abs_z=float(np.abs(x).max()))
        print(f'{role:15s} {x.shape}  label=1 share {lab.mean():.3f}')

    env_stats = {}
    for t in tasks:
        m = load_cache(cache, t['id'])[-1]
        k = (m['anomaly'] or {}).get('type', 'normal')
        e = env_stats.setdefault(k, dict(flights=0, redrawn=0, attempts_total=0))
        e['flights'] += 1
        e['redrawn'] += m.get('attempts', 1) > 1
        e['attempts_total'] += m.get('attempts', 1)
    units = {}
    chan_file = HERE.parent / 'data' / 'channels.json'
    if chan_file.exists():
        units = {r['name']: (r['unit'], r['meaning']) for r in json.loads(chan_file.read_text(encoding='utf-8'))}
    channels = []
    for nm in names:
        base = nm.split(':')[0]
        u, m = units.get(base, ('', ''))
        k = len(channels)
        channels.append(dict(name=nm, unit='' if ':' in nm else u, meaning=m,
                             bookkeeping=base.startswith(EXCLUDE_PREFIXES),
                             constant_in_normal=nm in const_in_normal,
                             normal_std_floored=nm in floored,
                             raw_mean=float(mean[k]), raw_std_normal=float(std_n[k]), scale_used=float(std[k]),
                             z_min=float(min(v['min'][k] for v in ch_stats.values())),
                             z_max=float(max(v['max'][k] for v in ch_stats.values()))))
    meta = dict(
        created=time.strftime('%Y-%m-%d %H:%M'), source='cruise_lab (JSBSim 1.3.1, B747)',
        window=dict(T=T_WIN, rate_hz=LOG_HZ, seconds=T_WIN / LOG_HZ, change_point_range=list(CP_RANGE)),
        flights=FLIGHTS, windows_per_flight=WIN_PER_FLIGHT, ramp_full_after_s=RAMP_S, flight_s=FLIGHT_S, profiles=list(PROFILES), severity_shrink_per_redraw=SEV_SHRINK,
        onset_range_s=list(ONSET_RANGE), hard_fraction_every_role=HARD_FRAC,
        envelope=dict(ENVELOPE, margin_s=ENVELOPE_MARGIN_S, max_attempts=MAX_ATTEMPTS,
                      rule='every flight is complete (FLIGHT_S) and inside the envelope until the last window '
                           'ends + margin (an exit later is recorded in envelope_exit_s); '
                           'otherwise the anomaly is redrawn with severity x SEV_SHRINK'),
        window_starts_s=list(WIN_STARTS_S), change_window_start_s=CHANGE_WIN_S,
        envelope_redraws=env_stats, clip_z=CLIP_Z if clip else None, duplicate_corr=DUP_CORR if dedup else None,
        constant_in_normal_dropped=drop_const, bookkeeping_kept=keep_bookkeeping,
        extra_noise_z=dict(std=noise_z, applied_to='all channels except sensor/* (which carry sensor noise)'),
        n_channels=len(names), circular_channels=sorted(circ), counts=counts,
        dropped={k: sorted(v) for k, v in dropped.items()},
        normal_std_floored=floored, channels=channels,
        normalisation='x = (raw - mean) / std, mean/std from train_normal; '
                      'std = max(std_normal, 0.1 * std_all_train, 1e-8)')
    (out / 'metadata.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')

    with open(out / 'flights.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['flight_id', 'seed', 'split', 'role', 'hard_normal', 'anomaly_type', 'group', 'variable',
                    'engine', 'severity', 'magnitude', 'unit', 'profile', 'sign', 'onset_s', 'end_s',
                    'flight_s', 'window_starts_s', 'envelope_exit_s', 'attempts'])
        for t in tasks:
            _, _, starts, _, _, meta = load_cache(cache, t['id'])
            a = meta['anomaly'] or {}
            w.writerow([t['id'], t['seed'], t['split'], t['role'], t['hard'] or '', a.get('type', ''),
                        a.get('group', ''), a.get('variable') or '', a.get('engine') or '',
                        a.get('severity', ''), a.get('magnitude', ''), a.get('unit', ''), a.get('profile', ''),
                        a.get('sign', ''), a.get('onset_s', ''), a.get('end_s', ''),
                        round(meta['n_rows'] / LOG_HZ, 1), ' '.join(f'{s / LOG_HZ:g}' for s in starts),
                        meta.get('envelope_exit_s') or '', meta.get('attempts', 1)])
    print(f'{len(names)} features ({len(keep)} channels, {len(circ)} as sin/cos); '
          f'{len(floored)} with floored normal std')


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE.parent / 'qcd_data'))
    ap.add_argument('--scale', type=float, default=1.0, help='multiply every flight count')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--assemble-only', action='store_true')
    ap.add_argument('--cache', default=None, help='flight cache (default <out>/cache); share one between outputs')
    ap.add_argument('--noise-z', type=float, default=0.0,
                    help='extra white noise (std, standardised units) on all non-sensor channels')
    ap.add_argument('--clip', action='store_true', help='clip standardised values to +-CLIP_Z')
    ap.add_argument('--dedup', action='store_true', help='drop duplicate channels (|corr| > DUP_CORR)')
    ap.add_argument('--drop-const', action='store_true', help='drop channels constant in normal flight')
    ap.add_argument('--drop-bookkeeping', action='store_true', help='drop clock/position bookkeeping channels')
    args = ap.parse_args()
    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    tasks = plan(args.scale)
    print(f'{len(tasks)} flights -> {out}')
    if not args.assemble_only:
        t0, done = time.time(), 0
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(run, t, cache): t['id'] for t in tasks}
            for f in as_completed(futs):
                fid, status, sec = f.result()
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    print(f'  {done}/{len(tasks)} flights  ({time.time() - t0:.0f} s)', flush=True)
    assemble(out, tasks, cache, args.noise_z, clip=args.clip, dedup=args.dedup, drop_const=args.drop_const,
             keep_bookkeeping=not args.drop_bookkeeping)


if __name__ == '__main__':
    main()
