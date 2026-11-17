"""Validate a dataset built by lab.qcd_dataset and write check_report.json.

    python -m lab.qcd_check [--data ./qcd_data]

Checks: shapes and finiteness, labels vs change points, no flight in two
splits, per-type balance, and a naive baseline (no training) that shows how
easy each anomaly type is. A naive rule catching everything would mean the
data is too easy; one catching nothing hints at an invisible fault.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROLES = ('train_normal', 'val_normal', 'test_normal', 'train_change', 'val_change', 'test_change')
CHANGE = ROLES[3:]


def naive_score(x):
    """Per window: time-average of the largest |z| over channels."""
    return np.abs(x).max(axis=-1).mean(axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(HERE.parent / 'qcd_data'))
    d = Path(ap.parse_args().data)
    rep, ok = {'checks': {}, 'baseline': {}}, True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        rep['checks'][name] = dict(passed=bool(cond), detail=detail)
        print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")

    z = {r: np.load(d / f'{r}.npz') for r in ROLES}
    meta = json.loads((d / 'metadata.json').read_text(encoding='utf-8'))
    D = {r: z[r]['x'].shape[-1] for r in ROLES}
    T = {r: z[r]['x'].shape[1] for r in ROLES}
    check('same T and D in every split', len(set(D.values())) == 1 and len(set(T.values())) == 1,
          f'T={T["train_normal"]} D={D["train_normal"]}')
    check('all values finite', all(np.isfinite(z[r]['x']).all() for r in ROLES))
    clip = meta.get('clip_z')
    if clip:
        check(f'all values within +-{clip:g}', all(np.abs(z[r]['x']).max() <= clip for r in ROLES))
    check('normal windows have label 0', all(z[r]['label'].sum() == 0 for r in ROLES if r.endswith('_normal')))
    for r in CHANGE:
        cp, lab = z[r]['change_point'], z[r]['label']
        check(f'{r}: label 0 before the change point, 1 from it to the end',
              all(lab[i, :cp[i]].sum() == 0 and lab[i, cp[i]:].min() == 1 for i in range(len(cp))),
              f'change point {cp.min()}..{cp.max()} of {lab.shape[1]}')
    ws = meta.get('window_starts_s')
    if ws:
        same = all(set(np.round(z[r]['window_start_s'], 3).tolist()) == ({meta['change_window_start_s']}
                   if r in CHANGE else set(ws)) for r in ROLES)
        check('every flight has the same window times', same,
              f"{ws} s (change windows at {meta['change_window_start_s']} s)")
        import csv
        with open(d / 'flights.csv', encoding='utf-8') as f:
            fl = {float(r['flight_s']) for r in csv.DictReader(f)}
        check('every flight is complete', fl == {meta['flight_s']}, f'{sorted(fl)} s')
    owner = {}
    for r in ROLES:
        for f in set(z[r]['flight_id'].tolist()):
            owner.setdefault(f, set()).add(r)
    check('no flight in two splits', all(len(v) == 1 for v in owner.values()), f'{len(owner)} flights')
    names = [c['name'] for c in meta['channels']]
    free = np.array([n not in set(meta['normal_std_floored']) for n in names])
    xn = z['train_normal']['x']
    if clip:                                                         # channels whose tail was clipped
        cl = (np.abs(xn) >= clip).mean(axis=(0, 1))
        rep['clip_affected_channels'] = {names[k]: round(float(cl[k]), 5) for k in np.where(cl > 1e-4)[0]}
        free &= cl <= 1e-4
    m, s = xn.mean(axis=(0, 1))[free], xn.std(axis=(0, 1))[free]     # floored / clipped channels excluded
    # clipping at +-clip_z trims the rare far tail, so allow a few % on the std
    check('train_normal is standardised (channels with >0.01 % clipped excluded)', np.abs(m).max() < 1e-2 and np.abs(s - 1).max() < 2e-2,
          f'max |mean| {np.abs(m).max():.1e}, std {s.min():.3f}..{s.max():.3f}; '
          f'{len(rep.get("clip_affected_channels", {}))} channels clip-affected (listed in report)')
    types = sorted(set(z['train_change']['anomaly_type'].tolist()))
    per = {r: {t: int((z[r]['anomaly_type'] == t).sum()) for t in types} for r in ROLES if not r.endswith('_normal')}
    check('every anomaly type equally often in every change split',
          all(min(v.values()) > 0 and len(set(v.values())) == 1 for v in per.values()), f'{len(types)} types')
    n_n = {r: len(z[r]['x']) for r in ROLES}
    check('normal and change samples balanced', all(n_n[f'{s}_normal'] == n_n[f'{s}_change'] for s in
          ('train', 'val', 'test')), json.dumps(n_n))
    rep['windows_per_type'] = per

    # naive baseline on the test split, on the last 30 s of each sample (always after the change point)
    tail = slice(int(z['test_change']['change_point'].max()) + 1 + 40, None)
    sn = naive_score(z['test_normal']['x'][:, tail])
    thr = float(np.quantile(sn, 0.95))
    sa, ty = naive_score(z['test_change']['x'][:, tail]), z['test_change']['anomaly_type']
    rep['baseline'] = dict(rule='last 30 s of the sample (after every change point): mean over time of max |z| '
                                'over channels; threshold = 95th pct of test_normal',
                           threshold=round(thr, 3), overall=round(float((sa > thr).mean()), 3),
                           per_type={t: round(float((sa[ty == t] > thr).mean()), 3) for t in types})
    print(f"  naive baseline at 5 % false alarms: {rep['baseline']['overall']:.2f} of change samples (after the change)")
    for t, v in rep['baseline']['per_type'].items():
        print(f'    {t:32s} {v:.2f}')
    rep['all_passed'] = bool(ok)
    (d / 'check_report.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')


if __name__ == '__main__':
    main()
