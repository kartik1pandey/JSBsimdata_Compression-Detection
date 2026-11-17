"""Per-run monitoring dashboards.

    python plot_runs.py                 # every finished run in results/ and results/ext/
    python plot_runs.py --live          # also runs still training (from results/**/live/*.jsonl)
    python plot_runs.py --only tf_lam0.25

For each run writes figs/runs/<tag>_training.png (every logged quantity against epoch, stages shaded),
<tag>_iters.png (per-iteration loss traces, runs started after the logging upgrade) and <tag>_eval.png
(LLR distributions, calibration, detection per fault type, post-change drift), plus figs/runs/INDEX.md.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT = Path(r'C:\Users\karti\IISc_data\qcd_spec_ckpt')
OUT = HERE / 'docs' / 'figs' / 'runs'
STAGE_C = {'s0': '#e8eef7', 's1': '#eef5ea', 's2': '#fbf0e4', 's3': '#f1ecf6'}
STAGE_N = {'s0': 'stage 0  reconstruct', 's1': 'stage 1  fit q0, q1', 's2': 'stage 2  alternate',
           's3': 'stage 3  refit q0, q1'}
INK, MUTED, BLUE, ORANGE, GREEN = '#222222', '#6b6b6b', '#2f6db5', '#d0603a', '#3a9b6b'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#bbbbbb', 'axes.grid': True, 'grid.color': '#e4e4e4',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9,
                     'xtick.color': MUTED, 'ytick.color': MUTED})

# (key, title, explanation shown under the title)
METRICS = [
    ('D', 'D̄  reconstruction error', 'MSE / σ²_X, lower = better copy of x'),
    ('L0', 'L0  NLL of q0 on normal', 'per number per step, train batches'),
    ('L1', 'L1  NLL of q1 on abnormal', 'per number per step, train batches'),
    ('L0val', 'L0 on val_normal', 'held-out check of q0 (overfit if it rises)'),
    ('E0s', 'E0[s]  drift on normal', 'must be < 0 (else false alarms)'),
    ('E1s', 'E1[s]  drift on abnormal', 'should be > 0 and large'),
    ('margin0', 'margin  E0[s] / sd', 'how far below 0 the normal drift sits'),
    ('J', 'J  task term (drift on D1)', 'what the encoder is pushed to raise'),
    ('j_ema', 'J_ema  smoothed J', 'denominator of the −log J gradient'),
    ('T', 'T̄ = −log J', 'the task loss term'),
    ('H', 'H̄  modellability', 'NLL of the code under q0/q1'),
    ('lam', 'λ schedule', 'ramps 0 → target over 5 epochs'),
    ('gn_fg', 'grad norm  encoder+decoder', 'before clipping at 1.0'),
    ('gn_q', 'grad norm  q0+q1', 'before clipping at 1.0'),
    ('lr_q', 'learning rate  q0+q1', 'cosine decay in the final fit'),
    ('floor_frac', 'σ at floor (fraction)', 'share of predicted σ stuck at σ_min'),
]


def load_runs(live, dirs=None):
    runs = {}
    for res in (dirs or (HERE / 'results', HERE / 'results' / 'ext')):
        for p in sorted(res.glob('*.json')):
            m = json.loads(p.read_text())
            if 'tag' in m:
                runs[m['tag']] = dict(m=m, log=m.get('log', []), res=res, done=True)
        if live:
            for p in sorted((res / 'live').glob('*.jsonl')):
                tag = p.stem
                if tag in runs:
                    continue
                rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
                args = rows[0] if rows and rows[0].get('stage') == 'args' else {}
                runs[tag] = dict(m=dict(tag=tag, args=args), log=[r for r in rows if r.get('stage') != 'args'],
                                 res=res, done=False)
    return runs


def title_of(m):
    a = m['args']
    if a.get('pipeline') == 'raw':
        return f"raw data, q = {a.get('q')}, seed {a.get('seed')}"
    dec = a.get('dec', 'conv')
    head = '' if a.get('head', 'gauss') == 'gauss' else f" ({a.get('mix')}-comp. MoG head)"
    out = '' if a.get('enc_out', 'ln') == 'ln' else ' (linear out)'
    return (f"encoder {a.get('enc')}{out} → k = {a.get('k')} → q0/q1 {a.get('q')}{head}, decoder {dec}, "
            f"λ = {a.get('lam'):g}, seed {a.get('seed')}")


def training_fig(tag, run, path):
    log = run['log']
    if not log:
        return False
    x = np.arange(len(log))
    stages = [r['stage'] for r in log]
    keys = [m for m in METRICS if any(r.get(m[0]) is not None for r in log)]
    nc = 4
    nr = int(np.ceil(len(keys) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(13, 2.6 * nr + 0.9), squeeze=False)
    for ax, (k, t, sub) in zip(axs.flat, keys):
        y = np.array([r.get(k) if r.get(k) is not None else np.nan for r in log], dtype=float)
        lo = 0
        for i in range(1, len(stages) + 1):
            if i == len(stages) or stages[i] != stages[lo]:
                ax.axvspan(lo - 0.5, i - 0.5, color=STAGE_C.get(stages[lo], '#f4f4f4'), lw=0)
                lo = i
        ax.plot(x, y, '-o', color=BLUE, lw=1.6, ms=2.5)
        if k in ('E0s', 'margin0'):
            ax.axhline(0, color=ORANGE, lw=1, ls='--')
        fin = y[np.isfinite(y)]
        if len(fin) and k in ('J', 'j_ema', 'E1s', 'E0s') and np.nanmax(np.abs(fin)) > 50:
            ax.set_yscale('symlog', linthresh=1)
        ax.set_title(t, loc='left', color=INK, pad=13)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7, color=MUTED, va='bottom', ha='left')
        ax.set_xlim(-0.5, len(log) - 0.5)
    for ax in list(axs.flat)[len(keys):]:
        ax.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_C[s]) for s in STAGE_N if s in stages]
    fig.legend(handles, [STAGE_N[s] for s in STAGE_N if s in stages], loc='lower center', ncol=4, frameon=False)
    fig.suptitle(f"Training monitor — {title_of(run['m'])}" + ('' if run['done'] else '   (STILL TRAINING)'),
                 x=0.01, ha='left', fontsize=11)
    fig.supxlabel('epoch (all stages in order)', y=0.045, fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


LOSS_PANELS = [
    (('D',), 'val_D', 'D̄  reconstruction error', 'MSE / σ²_X, both classes'),
    (('L0',), 'val_L0', 'L0  q0 NLL on normal', 'per number per step'),
    (('L1',), 'val_L1', 'L1  q1 NLL on abnormal', 'per number per step'),
    (('H', 'Hq'), 'val_H', 'H = L0 + L1  modellability', 'NLL of the code under q0/q1'),
    (('J',), 'val_J', 'J  drift on abnormal (t > t0)', 'what the task term raises'),
    (('T',), 'val_T', 'T = −log J  task loss', ''),
    (('loss',), 'val_loss', 'combined  L = (1−λ)D̄ + λT + ηH', 'at the λ of that epoch'),
    ((), 'z_std', 'code scale  (mean std of z)', 'validation codes'),
    ((), 'pi_max', 'largest mixture weight of q0', '1 = one component does everything'),
]


def losses_fig(tag, run, path):
    """Train (solid) vs validation (dashed) for every loss term, individually and combined, per epoch."""
    log = run['log']
    if not log or not any('val_L0' in r for r in log):
        return False
    x = np.arange(len(log))
    stages = [r['stage'] for r in log]
    panels = [pp for pp in LOSS_PANELS if any(r.get(pp[1]) is not None for r in log)]
    nc = 3
    nr = int(np.ceil(len(panels) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(13, 2.7 * nr + 1.0), squeeze=False)
    for ax, (tk, vk, t, sub) in zip(axs.flat, panels):
        lo = 0
        for i in range(1, len(stages) + 1):
            if i == len(stages) or stages[i] != stages[lo]:
                ax.axvspan(lo - 0.5, i - 0.5, color=STAGE_C.get(stages[lo], '#f4f4f4'), lw=0)
                lo = i
        yt = np.array([next((r[k] for k in tk if r.get(k) is not None), np.nan) for r in log], dtype=float)
        yv = np.array([r.get(vk) if r.get(vk) is not None else np.nan for r in log], dtype=float)
        if np.isfinite(yt).any():
            ax.plot(x, yt, '-o', color=BLUE, lw=1.6, ms=2.5, label='train')
        ax.plot(x, yv, '--s', color=ORANGE, lw=1.4, ms=2.5, label='validation')
        fin = np.concatenate([yt[np.isfinite(yt)], yv[np.isfinite(yv)]])
        if len(fin) and tk and tk[0] == 'J' and np.nanmax(np.abs(fin)) > 50:
            ax.set_yscale('symlog', linthresh=1)
        ax.set_title(t, loc='left', color=INK, pad=13)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7, color=MUTED, va='bottom', ha='left')
        ax.set_xlim(-0.5, len(log) - 0.5)
    axs.flat[0].legend(frameon=False, fontsize=8)
    for ax in list(axs.flat)[len(panels):]:
        ax.axis('off')
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_C[s]) for s in STAGE_N if s in stages]
    fig.legend(handles, [STAGE_N[s] for s in STAGE_N if s in stages], loc='lower center', ncol=4, frameon=False)
    fig.suptitle(f"Losses, train vs validation — {title_of(run['m'])}" + ('' if run['done'] else '   (STILL TRAINING)'),
                 x=0.01, ha='left', fontsize=11)
    fig.supxlabel('epoch (all stages in order)', y=0.045, fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def iters_fig(tag, run, path):
    p = run['res'] / 'live' / f'{tag}_trace.npz'
    if not p.exists():
        return False
    tr = dict(np.load(p))
    keys = [k for k in ('s0_D', 's1_L0', 's1_L1', 's2_loss', 's2_D', 's2_H', 's2_J', 's2_J_ema', 's3_L0', 's3_L1')
            if k in tr and len(tr[k])]
    if not keys:
        return False
    nc = 4
    nr = int(np.ceil(len(keys) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(13, 2.2 * nr + 0.6), squeeze=False)
    for ax, k in zip(axs.flat, keys):
        y = tr[k].astype(float)
        ax.plot(y, color='#9bb7dd', lw=0.6)
        w = max(1, len(y) // 40)
        if len(y) > w:
            ax.plot(np.arange(w - 1, len(y)), np.convolve(y, np.ones(w) / w, 'valid'), color=BLUE, lw=1.6)
        if k.endswith('J') or k.endswith('J_ema') or k == 's2_loss':
            if np.nanmax(np.abs(y)) > 50:
                ax.set_yscale('symlog', linthresh=1)
        ax.set_title(k.replace('_', '  ', 1), loc='left')
    for ax in list(axs.flat)[len(keys):]:
        ax.axis('off')
    fig.suptitle(f"Per-iteration traces (light: every step, dark: moving mean) — {title_of(run['m'])}",
                 x=0.01, ha='left', fontsize=10.5)
    fig.supxlabel('iteration within the stage', fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def slog(v):
    return np.sign(v) * np.log10(1 + np.abs(v))


def eval_fig(tag, run, path):
    m = run['m']
    p = run.get('ckpt', CKPT) / f'{tag}_llr.npz'
    if not run['done'] or not p.exists():
        return False
    L = np.load(p)
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.6))
    # 1 per-step LLR distributions
    ax = axs[0, 0]
    bins = np.linspace(-4, 4, 81)
    for r, c, lab in (('test_normal', BLUE, 'normal flights'), ('test_abnormal', ORANGE, 'abnormal flights')):
        ax.hist(slog(L[r][:, 40:].ravel()), bins=bins, color=c, alpha=0.55, density=True, label=lab)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title('Per-step evidence s_t = log q1 − log q0', loc='left')
    ax.set_xlabel('sign · log10(1 + |s_t|)   (right = "looks abnormal")')
    ax.legend(frameon=False)
    # 2 calibration
    ax = axs[0, 1]
    import spec_run as S
    M = np.sort(S.cusum(L['val_normal']).max(1))
    h = m['h']
    ax.plot(np.arange(1, len(M) + 1), slog(M), 'o', color=BLUE, ms=3)
    ax.axhline(slog(h), color=ORANGE, lw=1.4)
    ax.text(2, slog(h), f'  h = {h:.3g}', color=ORANGE, va='bottom')
    ax.set_title('Calibration: max CUSUM of each normal val flight', loc='left')
    ax.set_xlabel('rank (sorted)')
    ax.set_ylabel('log10(1 + max W)')
    # 3 FA / detection summary
    ax = axs[0, 2]
    d = m['detect']
    vals = [m['FA_test_normal'], d['FA_pre'], d['DR'], d['miss']]
    labs = ['false alarm\nnormal flights', 'alarm before\nchange', 'detected', 'missed']
    cols = ['#b5b5b5', '#b5b5b5', GREEN, '#d9d9d9']
    ax.bar(labs, vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{100 * v:.0f}%', ha='center', fontsize=8)
    ax.axhline(0.05, color=ORANGE, lw=1, ls='--')
    ax.set_ylim(0, 1.05)
    med = d['ADD_median_s']
    ax.set_title(f"Outcome on test flights (median delay {med:.0f} s)" if med is not None
                 else 'Outcome on test flights', loc='left')
    # 4 per type
    ax = axs[1, 0]
    per = m.get('detect_per_type', {})
    ty = sorted(per, key=lambda t: per[t]['DR'])
    ax.barh(ty, [per[t]['DR'] for t in ty], color=GREEN, height=0.6)
    ax.set_xlim(0, 1)
    ax.set_title('Detection rate per fault type (10 flights each)', loc='left')
    ax.tick_params(axis='y', labelsize=7)
    # 5 drift curve
    ax = axs[1, 1]
    cur = m.get('drift_curve_val_change', {})
    if cur:
        xx = np.array([int(k) for k in cur]) / 4.0
        yy = np.array(list(cur.values()))
        o = np.argsort(xx)
        ax.plot(xx[o], slog(yy[o]), '-o', color=BLUE, ms=3, lw=1.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title('Mean evidence around the change (val_change)', loc='left')
    ax.set_xlabel('seconds since change')
    ax.set_ylabel('sign · log10(1 + |E s_t|)')
    # 6 cross-entropy table
    ax = axs[1, 2]
    ax.axis('off')
    t = m['test']
    rows = [('CE00  q0 on normal', t['CE00']), ('CE01  q1 on normal', t['CE01']),
            ('CE10  q0 on abnormal', t['CE10']), ('CE11  q1 on abnormal', t['CE11']),
            ('K = CE10 − CE11', t['K']), ('E0[s] (want < 0)', t['E0s']), ('E1[s] (want > 0)', t['E1s']),
            ('sign check', 'pass' if t['signs_ok'] else 'FAIL')]
    if 'D_test_normal' in m:
        rows.insert(0, ('D̄ normal / abnormal', f"{m['D_test_normal']:.3f} / {m['D_test_abnormal']:.3f}"))
    for i, (k, v) in enumerate(rows):
        ax.text(0, 1 - i * 0.105, k, fontsize=9, color=INK, va='top')
        ax.text(0.98, 1 - i * 0.105, v if isinstance(v, str) else f'{v:.3g}', fontsize=9, ha='right', va='top',
                color=(ORANGE if v == 'FAIL' else INK))
    ax.set_title('Diagnostics on pure test sets (nats / step)', loc='left')
    fig.suptitle(f"Evaluation — {title_of(m)}", x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--res', nargs='*', help='result dirs (default results, results/ext)')
    ap.add_argument('--ckpt', default=None, help='checkpoint dir of those runs')
    ap.add_argument('--out', default=None, help='output dir for the pngs')
    a = ap.parse_args()
    global OUT
    OUT = Path(a.out) if a.out else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.live, [Path(r) for r in a.res] if a.res else None)
    if a.ckpt:
        for r in runs.values():
            r['ckpt'] = Path(a.ckpt)
    idx = ['# Per-run dashboards (auto-generated by plot_runs.py)', '',
           'Each run has a training monitor (every logged quantity per epoch, stages shaded), per-iteration '
           'traces (runs started after the logging upgrade) and an evaluation sheet.', '']
    for tag, run in runs.items():
        if a.only and a.only not in tag:
            continue
        made = []
        for kind, fn in (('training', training_fig), ('losses', losses_fig), ('iters', iters_fig), ('eval', eval_fig)):
            if fn(tag, run, OUT / f'{tag}_{kind}.png'):
                made.append(kind)
        if made:
            idx.append(f"## {tag}{'' if run['done'] else ' (still training)'}")
            idx.append(title_of(run['m']))
            idx += [f'![{k}]({tag}_{k}.png)' for k in made] + ['']
        print(tag, made)
    if not a.only:
        (OUT / 'INDEX.md').write_text('\n'.join(idx) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
