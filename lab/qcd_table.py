"""Write DATASET_DETAILS.md and channels.csv for a dataset built by lab.qcd_dataset.

    python -m lab.qcd_table [--data ./qcd_data]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import anomalies as an

HERE = Path(__file__).parent
ROLES = ('train_normal', 'val_normal', 'test_normal', 'train_change', 'val_change', 'test_change')
ROLE_TEXT = {
    'train_normal': 'normal flight', 'val_normal': 'normal flight', 'test_normal': 'normal flight',
    'train_change': 'normal, then the anomaly starts inside the sample',
    'val_change': 'normal, then the anomaly starts inside the sample',
    'test_change': 'normal, then the anomaly starts inside the sample',
}


def md(rows, head):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(HERE.parent / 'qcd_data'))
    d = Path(ap.parse_args().data)
    meta = json.loads((d / 'metadata.json').read_text(encoding='utf-8'))
    with open(d / 'flights.csv', encoding='utf-8') as f:
        fl = list(csv.DictReader(f))
    W = meta['window']
    T, hz = W['T'], W['rate_hz']
    D = meta['n_channels']
    ws = meta['window_starts_s']
    cw = meta['change_window_start_s']
    horizon = max(ws) + T / hz + meta['envelope']['margin_s']
    size = {r: (d / f'{r}.npz').stat().st_size / 2 ** 20 for r in ROLES}
    c = meta['counts']
    L = [f"# Dataset details: `{d.name}`", '',
         f"Built {meta['created']} from {meta['source']}.", '']

    # 1. overview
    n_win = sum(c[r]['windows'] for r in ROLES)
    L += ['## 1. Overview', '', md([
        ['Aircraft / simulator', 'Boeing 747, JSBSim 1.3.1, autopilot + autothrottle in cruise'],
        ['Sampling rate', f'{hz:g} Hz'],
        ['Window length', f'T = {T} samples = {T / hz:g} s'],
        ['Channels per sample', f'D = {D}'],
        ['Array shape per file', '(N, T, D) float32, plus label (N, T)'],
        ['Flight length', f"{meta['flight_s']:g} s, every flight"],
        ['Sample time (every flight, normal and change)',
         ', '.join(f'{s:g}-{s + T / hz:g} s' for s in ws) + ' - one sample per flight, identical in every flight'],
        ['Kinds of sample', 'normal (whole sample normal) and change (normal, then the anomaly from the change '
                            'point to the end of the flight); no fully-abnormal samples'],
        ['Change point (change flights)', f"sample {W['change_point_range'][0]}-{W['change_point_range'][1]} of the "
                                          f"window = onset {cw + W['change_point_range'][0] / hz:g}-"
                                          f"{cw + W['change_point_range'][1] / hz:g} s"],
        ['Flights', f'{len(fl)}'],
        ['Windows', f'{n_win}'],
        ['Anomaly types', f"{len(an.CATALOG)} (equal flights and windows per type)"],
        ['Profiles', ', '.join(meta['profiles'])],
        ['Hard conditions', f"{meta['hard_fraction_every_role'] * 100:g} % of flights in every role "
                            '(turbulence, strong wind, hot day, cold day, heavy), normal and change alike'],
        ['Normalisation', meta['normalisation']],
        ['Clipping', 'none' if meta['clip_z'] is None else f"+-{meta['clip_z']}"],
        ['Duplicate channels removed', 'no' if meta['duplicate_corr'] is None else f"|corr| > {meta['duplicate_corr']}"],
        ['Extra noise', 'none' if not meta['extra_noise_z']['std'] else f"{meta['extra_noise_z']['std']} std"],
        ['Total size', f'{sum(size.values()):.0f} MB (uncompressed .npz)'],
    ], ['Item', 'Value']), '']

    # 2. files
    hard = {}
    for r in fl:
        hard.setdefault(r['role'], {})
        if r['hard_normal']:
            hard[r['role']][r['hard_normal']] = hard[r['role']].get(r['hard_normal'], 0) + 1
    rows = []
    for r in ROLES:
        k = c[r]
        rows.append([f'`{r}.npz`', ROLE_TEXT[r], k['flights'], k['windows'], f"({', '.join(map(str, k['shape']))})",
                     f"{k['label_fraction']:.3f}", sum(hard.get(r, {}).values()),
                     f"{k['max_abs_z']:.1f}", f"{k['fraction_beyond_10'] * 100:.3f} %", f'{size[r]:.0f} MB'])
    tot = ['**total**', '', sum(c[r]['flights'] for r in ROLES), n_win, '', '', sum(sum(h.values()) for h in hard.values()),
           '', '', f'{sum(size.values()):.0f} MB']
    L += ['## 2. Files', '', md(rows + [tot], ['File', 'Content', 'Flights', 'Windows', 'Shape (N, T, D)',
                                              'Label = 1 share', 'Hard-condition flights', 'Max |z|',
                                              'Values with |z| > 10', 'Size']), '',
          f"Each flight gives {meta['windows_per_flight']} window(s); change flights 1. Every file also holds "
          '`label (N, T)`, `flight_id`, `anomaly_type`, `group`, `variable`, `severity`, `profile`, '
          '`window_start_s`; the change files also `change_point`.', '']

    # 3. normal vs change balance
    nn = sum(c[r]['windows'] for r in ROLES if r.endswith('_normal'))
    na = sum(c[r]['windows'] for r in ROLES if r.endswith('_change'))
    L += ['## 3. Normal vs change', '', md([
        [s, c[f'{s}_normal']['windows'], c[f'{s}_change']['windows']] for s in ('train', 'val', 'test')]
        + [['**total**', nn, na]], ['Split', 'Normal samples', 'Change samples']), '']

    # 4. per anomaly type
    rows = []
    for t, cat in an.CATALOG.items():
        ff = [r for r in fl if r['anomaly_type'] == t]
        sev = np.array([float(r['severity']) for r in ff])
        on = np.array([float(r['onset_s']) for r in ff])
        prof = {p: sum(r['profile'] == p for r in ff) for p in meta['profiles']}
        var = sorted({r['variable'] or (f"eng {r['engine']}" if r['engine'] else '') for r in ff})
        red = sum(int(r['attempts']) > 1 for r in ff)
        late = sum(bool(r['envelope_exit_s']) for r in ff)
        per = [c[r]['per_type'].get(t, 0) for r in ROLES[3:]]
        rows.append([t, cat['group'], ', '.join(var), *per, sum(per), f'{sev.min():.2f}-{sev.max():.2f} ({sev.mean():.2f})',
                     ' / '.join(str(prof[p]) for p in meta['profiles']),
                     f'{on.min():.0f}-{on.max():.0f}', red, late])
    L += ['## 4. Anomaly types', '',
          md(rows, ['Type', 'Group', 'Variables used', 'Train samples', 'Val samples', 'Test samples', 'Total',
                    'Severity min-max (mean)', ' / '.join(meta['profiles']) + ' flights', 'Onset s (flight time)',
                    'Flights redrawn milder', f'Leave envelope after {horizon:g} s']), '',
          'Per type: ' + ', '.join(f"{v} {k}" for k, v in meta['flights']['change_per_type'].items())
          + ' change flights, one sample each. '
          '"Redrawn milder": the first draw drove the aircraft out of the flight envelope before the last window '
          f'ended, so the anomaly was redrawn with severity x 0.7. "Leave envelope after {horizon:g} s": the aircraft '
          'leaves the envelope only after the data used (flight still simulated to 1500 s).', '']
    txt = {t: c_['text'].replace(' (capped at 120 s)', '') for t, c_ in an.CATALOG.items()}
    L += ['### What each type does', '', 'In this dataset every anomaly lasts from its onset to the end of the '
          'flight (no per-type duration cap).', '',
          md([[t, c_['formula'], txt[t]] for t, c_ in an.CATALOG.items()],
                                             ['Type', 'Formula', 'Effect']), '']

    # 5. hard conditions
    conds = sorted({h for v in hard.values() for h in v})
    L += ['## 5. Hard conditions per role', '',
          md([[r, *[hard.get(r, {}).get(h, 0) for h in conds], sum(hard.get(r, {}).values()), c[r]['flights']]
              for r in ROLES], ['Role', *conds, 'Hard total', 'Flights']), '']

    # 6. channels
    ch = meta['channels']
    groups = {}
    for x in ch:
        g = x['name'].split('/')[0]
        groups[g] = groups.get(g, 0) + 1
    L += ['## 6. Channels', '',
          f"{D} channels = every channel that changes inside the training windows. Of the 680 logged JSBSim "
          f"properties, {len(meta['dropped']['not changing'])} never change in any training window and are left out; "
          f"nothing else is removed. {len(meta['circular_channels'])} heading-type angles are stored as sin/cos "
          f"pairs (so 359 deg -> 0 deg is not a jump). The full list with units, meaning and value ranges is in "
          f"`channels.csv`.", '',
          md(sorted(groups.items(), key=lambda kv: -kv[1]), ['Group (property prefix)', 'Channels']), '',
          md([['Bookkeeping (clock, position, orbital)', sum(x['bookkeeping'] for x in ch),
               'kept; marked `bookkeeping` in channels.csv'],
              ['Constant in normal flight, change only under a fault', sum(x['constant_in_normal'] for x in ch),
               ', '.join(x['name'] for x in ch if x['constant_in_normal'])],
              ['Std floored (nearly constant in normal flight)', sum(x['normal_std_floored'] for x in ch),
               'scale = 0.1 x all-train std'],
              ['sensor/* (with measurement noise)', sum(x['name'].startswith('sensor/') for x in ch), ''],
              ['Contain NaN', len(meta['dropped']['contains NaN']), '']],
             ['Kind', 'Channels', 'Note']), '']
    L += ['## 7. Processing', '', md([
        ['1', 'Fly every flight for 1500 s at 4 Hz with JSBSim (anomaly injected from the onset to the end)'],
        ['2', 'Check the flight envelope up to the end of the last window + 10 s (bank 45 deg, pitch -15..25 deg, '
              'altitude error 3000 ft, Mach 0.92, CAS 160 kt); if left, redraw the anomaly milder (max 8 tries)'],
        ['3', 'Cut the windows at the fixed times'],
        ['4', 'Keep every channel that changes; headings to sin/cos'],
        ['5', 'z-score with train_normal mean and std (std floored for nearly-constant channels); '
              'mean/std saved in normalization.npz so raw values are x * std + mean'],
        ['-', 'No clipping, no duplicate removal, no extra noise, no filtering'],
    ], ['Step', 'What']), '']
    (d / 'DATASET_DETAILS.md').write_text('\n'.join(L) + '\n', encoding='utf-8')

    with open(d / 'channels.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['index', 'name', 'unit', 'meaning', 'bookkeeping', 'constant_in_normal', 'std_floored',
                    'raw_mean_normal', 'raw_std_normal', 'scale_used', 'z_min_all_files', 'z_max_all_files'])
        for i, x in enumerate(ch):
            w.writerow([i, x['name'], x['unit'], x['meaning'], int(x['bookkeeping']), int(x['constant_in_normal']),
                        int(x['normal_std_floored']), f"{x['raw_mean']:.6g}", f"{x['raw_std_normal']:.6g}",
                        f"{x['scale_used']:.6g}", f"{x['z_min']:.3f}", f"{x['z_max']:.3f}"])
    print(f"wrote {d / 'DATASET_DETAILS.md'} and {d / 'channels.csv'}")


if __name__ == '__main__':
    main()
