"""Extra figures for the presentation: architecture diagrams (every block, sizes, parameter counts), the loss and
training flow, and results across every variation (family, λ, code size k, head, encoder output, anomaly type).

Reads only result JSONs (results/, results/ext, results/srv, results/mog); parameter counts are taken by building
the models of models.py, so nothing is typed in by hand. Writes docs/figs/more/*.png and prints the numbers the
figures show, so docs/FIGURES.md can quote them.
    python make_more_figures.py
"""
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import models  # noqa: E402

HERE = Path(__file__).parent
R = HERE / 'results'
OUT = HERE / 'docs' / 'figs' / 'more'
OUT.mkdir(parents=True, exist_ok=True)
D_IN, HZ, T0 = 297, 4, 40

INK, MUTED, GRID = '#1f2933', '#6b7280', '#e5e7eb'
FAM = {'tf': ('Transformer', '#2f6db5'), 'gru': ('GRU', '#0f8a6a'), 'tcn': ('TCN', '#c2410c'),
       'lin': ('Linear AR(4)', '#6b7280')}
MOGC = '#7a3fb8'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#9ca3af', 'axes.grid': True, 'grid.color': GRID,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlelocation': 'left',
                     'axes.titlesize': 10, 'figure.facecolor': 'white'})


# ------------------------------------------------------------------ data
def load_all():
    runs = []
    for d, src in ((R, 'laptop'), (R / 'ext', 'laptop'), (R / 'srv', 'server'), (R / 'mog', 'server')):
        for p in sorted(d.glob('*.json')):
            try:
                m = json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                continue
            if 'detect' not in m or 'args' not in m:
                continue
            a = dict(m['args'])
            a.setdefault('dec', 'conv')
            a.setdefault('head', 'gauss')
            a.setdefault('enc_out', 'ln')
            m['a'], m['src'] = a, src
            runs.append(m)
    return runs


RUNS = load_all()


def sel(**kw):
    out = []
    for m in RUNS:
        a = m['a']
        if all(a.get(k) == v for k, v in kw.items()):
            out.append(m)
    return out


def std_cae(fam, k=8, lam=0.0, **kw):
    return sel(pipeline='cae', enc=fam, q=fam, k=k, lam=lam, dec='conv', head='gauss', enc_out='ln', **kw)


def mog(k=8, lam=0.0, out='linear'):
    return sel(pipeline='cae', head='mog', k=k, lam=lam, enc_out=out)


def mean(rs, f):
    v = [f(m) for m in rs]
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def box(ax, x, y, w, h, text, fc, ec=None, fs=8, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.015', fc=fc,
                                ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc,
            fontweight='bold' if bold else 'normal', linespacing=1.35)


def arrow(ax, p, q, c=INK, ls='-', rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, color=c, lw=1.2, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}'))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax


def nparam(mod):
    return sum(p.numel() for p in mod.parameters())


def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


# ------------------------------------------------------------------ A1 whole pipeline
def a1_pipeline(k=8):
    f = models.Encoder('tf', D_IN, k, out='linear')
    g = models.make_decoder('conv', D_IN, k)
    q = models.make_q('mog', 'tf', k)
    qg = models.make_q('gauss', 'tf', k)
    P = dict(f=nparam(f), g=nparam(g), q=nparam(q), qg=nparam(qg))
    fig, ax = blank((15, 7.2))
    ax.text(0, 0.985, f'A1  Full system (supervisor spec, k = {k}): what runs on the aircraft, what is sent, what '
            'runs on the ground', fontsize=12, fontweight='bold', va='top')
    # aircraft side
    ax.add_patch(FancyBboxPatch((0.0, 0.08), 0.43, 0.82, boxstyle='round,pad=0.005', fc='#f3f6fb', ec='#c7d2e3'))
    ax.text(0.01, 0.875, 'ON THE AIRCRAFT  (causal: uses only the past)', fontsize=8.5, color='#2f6db5',
            fontweight='bold')
    ax.add_patch(FancyBboxPatch((0.55, 0.08), 0.45, 0.82, boxstyle='round,pad=0.005', fc='#f7f3fb', ec='#d9cbe8'))
    ax.text(0.56, 0.875, 'ON THE GROUND  (receives only z)', fontsize=8.5, color=MOGC, fontweight='bold')
    box(ax, 0.015, 0.62, 0.11, 0.2, f'x_t\n{D_IN} channels\n4 Hz\n(normalised)', '#ffffff', '#9ca3af')
    enc = ('Encoder f  (causal transformer)\n\n'
           f'Linear {D_IN} → 64\n+ sinusoidal position\n'
           '2 × pre-LN transformer layer\n  4 heads, width 64, FF 128\n  banded mask: last 128 steps\n'
           f'Linear 64 → k = {k}\noutput: linear (spec) or LayerNorm\n\n'
           f'{P["f"]:,} parameters')
    box(ax, 0.16, 0.40, 0.25, 0.44, enc, '#dbe7f7', '#2f6db5', fs=8)
    arrow(ax, (0.125, 0.72), (0.16, 0.72))
    box(ax, 0.455, 0.62, 0.075, 0.2, f'z_t\nk = {k}\nnumbers\nper step', '#fde7d4', '#c2410c', bold=True)
    arrow(ax, (0.41, 0.72), (0.455, 0.72))
    ax.text(0.492, 0.86, 'radio link', ha='center', fontsize=8, color='#c2410c')
    # ground side
    q_txt = ('q0 (normal)  and  q1 (abnormal)\n'
             'causal transformer, context 128\n'
             'same backbone as f, input k\n'
             'head: mixture of 4 Gaussians,\n'
             'covariance  W Wᵀ + diag(d²), rank 8\n\n'
             f'{P["q"]:,} parameters each\n(old diagonal head: {P["qg"]:,})')
    box(ax, 0.58, 0.50, 0.22, 0.34, q_txt, '#ebe2f5', MOGC, fs=8)
    arrow(ax, (0.53, 0.72), (0.58, 0.70))
    cus = ('s_t = log q1(z_t) − log q0(z_t)\n\nCUSUM\nS_t = max(0, S_(t−1) + s_t)\n\nalarm when S_t > h\n'
           'h set on val_normal\nfor 5 % false alarms')
    box(ax, 0.83, 0.50, 0.16, 0.34, cus, '#fff4d6', '#b5862f', fs=8)
    arrow(ax, (0.80, 0.67), (0.83, 0.67))
    dec = ('Decoder g  (training + reconstruction only,\nnot in the detection path; non-causal)\n\n'
           f'Conv1d k → 128, kernel 5, GELU\nConv1d 128 → 128, kernel 5, GELU\nConv1d 128 → {D_IN}, kernel 1\n\n'
           f'{P["g"]:,} parameters')
    box(ax, 0.58, 0.12, 0.22, 0.3, dec, '#e3f1ec', '#0f8a6a', fs=8)
    arrow(ax, (0.53, 0.66), (0.62, 0.42), rad=0.15)
    box(ax, 0.83, 0.18, 0.16, 0.18, 'x̂_t  (297 channels)\n\nD̄ = mean (x − x̂)² / σ²_x', '#ffffff', '#0f8a6a',
        fs=8)
    arrow(ax, (0.80, 0.27), (0.83, 0.27))
    # training note
    ax.text(0.01, 0.30, 'Trained jointly with\nL = (1 − λ) D̄ + λ T + η H\n(see A4 for every term)\n\n'
            f'Sizes at k = {k}:\n f {P["f"]:,}   g {P["g"]:,}\n q0 = q1 {P["q"]:,}\n'
            f' total {P["f"] + P["g"] + 2 * P["q"]:,}', fontsize=8.5, va='top', color=INK)
    ax.text(0.01, 0.02, f'Compression: {D_IN} numbers → {k} numbers per step ({D_IN / k:.0f}× fewer).',
            fontsize=8.5, color=MUTED)
    save(fig, 'A1_pipeline.png')
    return P


# ------------------------------------------------------------------ A2 MoG head
def a2_mog(k=8, M=4, r=8):
    fig, ax = blank((14, 5.6))
    n_out = M * (1 + 2 * k + k * r)
    ax.text(0, 0.98, f'A2  Mixture-of-multivariate-Gaussians head of q0 / q1 (k = {k}, {M} components, rank {r})',
            fontsize=12, fontweight='bold', va='top')
    box(ax, 0.0, 0.42, 0.11, 0.22, 'z_1 … z_t\n(last 128 codes)', '#fde7d4', '#c2410c')
    box(ax, 0.14, 0.38, 0.17, 0.3, 'causal transformer\n2 layers, width 64\n4 heads, mask 128\n\n'
        f'Linear 64 → {n_out}', '#ebe2f5', MOGC)
    arrow(ax, (0.11, 0.53), (0.14, 0.53))
    parts = [('logits  (M = 4)', f'{M}', 'π_m = softmax(logits)'),
             ('mean residual δ_m', f'{M} × {k}', 'μ_m = z_t + δ_m  (persistence + residual)'),
             ('scale d_m', f'{M} × {k}', 'd_m = σ_min + softplus(·)  (floor σ_min)'),
             ('factor W_m', f'{M} × {k} × {r}', f'W_m scaled by 0.5/√{r}')]
    for i, (nm, sz, rule) in enumerate(parts):
        y = 0.78 - i * 0.19
        box(ax, 0.36, y, 0.16, 0.13, f'{nm}\n{sz} numbers', '#f5f0fa', MOGC, fs=8)
        arrow(ax, (0.31, 0.53), (0.36, y + 0.065), c=MUTED)
        ax.text(0.535, y + 0.065, rule, fontsize=8.5, va='center')
    box(ax, 0.78, 0.36, 0.215, 0.34, 'q(z_(t+1) | z_≤t) =\nΣ_m π_m · N(z_(t+1); μ_m, Σ_m)\n\n'
        'Σ_m = W_m W_mᵀ + diag(d_m²)\n\ncorrelated dimensions,\nmulti-modal forecasts', '#fff4d6', '#b5862f',
        fs=8.5)
    arrow(ax, (0.745, 0.53), (0.78, 0.53), c=MUTED)
    ax.text(0.0, 0.08, f'Per step the backbone outputs {M}·(1 + 2k + k·r) = {n_out} numbers. A dense k × k Cholesky '
            f'factor per component would need k(k+1)/2 = {k * (k + 1) // 2} (k = 200: 20,100), so the covariance is '
            'low-rank + diagonal.\nlog q is computed with torch LowRankMultivariateNormal + MixtureSameFamily; if the '
            'float32 Cholesky of the k×r capacitance matrix fails, the same distribution is rebuilt in float64.',
            fontsize=8.5, color=MUTED, va='bottom')
    save(fig, 'A2_mog_head.png')


# ------------------------------------------------------------------ A3 the four families
def a3_families(k=8):
    fig, axs = plt.subplots(1, 4, figsize=(16, 5.4))
    desc = {
        'tf': ['Linear in → 64', '+ sinusoidal position', 'pre-LN transformer layer ×2',
               '  self-attention, 4 heads', '  banded causal mask (128)', '  FF 64→128→64, GELU', 'Linear 64 → out'],
        'gru': ['Linear in → 64, GELU', 'GRU, hidden 64', '  (whole history,', '   no finite context)',
                'Linear 64 → out'],
        'tcn': ['Conv1d 1×1 in → 64', 'causal conv k=2, dil 1', '  … dil 2, 4, 8, 16, 32', 'causal conv k=2, dil 64',
                '  residual + GELU each', '  receptive field = 128', 'Conv1d 1×1 64 → out'],
        'lin': ['μ = z_t + A [z_t … z_(t−3)]', 'A: Linear 4k → k', '(zero-initialised)', 'σ: one learned value',
                '   per dimension', '(classical VAR(4) baseline;', ' q only, no encoder)']}
    for ax, fam in zip(axs, ('tf', 'gru', 'tcn', 'lin')):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        nm, c = FAM[fam]
        ax.text(0.5, 0.97, nm, ha='center', fontsize=11, fontweight='bold', color=c, va='top')
        for i, t in enumerate(desc[fam]):
            ax.text(0.06, 0.86 - i * 0.075, t, fontsize=8.5, va='top', family='monospace')
        rows = []
        if fam != 'lin':
            rows.append(('encoder f (k=8)', nparam(models.Encoder(fam, D_IN, k))))
        rows.append(('q, diag. Gaussian', nparam(models.make_q('gauss', fam, k))))
        if fam != 'lin':
            rows.append(('q, MoG (4, rank 8)', nparam(models.make_q('mog', fam, k))))
            rows.append(('q on raw data (k=297)', nparam(models.make_q('gauss', fam, D_IN))))
        for i, (a, b) in enumerate(rows):
            ax.text(0.06, 0.26 - i * 0.06, f'{a:<22}{b:>9,}', fontsize=8.5, family='monospace', va='top')
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.9, boxstyle='round,pad=0.005', fc='none', ec=c, lw=1.2))
    fig.suptitle('A3  The four model families (same backbone is used for the encoder f and for q0 / q1); '
                 'parameter counts from models.py', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, 'A3_families.png')


# ------------------------------------------------------------------ A4 loss terms
def a4_losses():
    fig, ax = blank((15, 6.4))
    ax.text(0, 0.98, 'A4  Every loss term, what it measures, and which parameters it trains', fontsize=12,
            fontweight='bold', va='top')
    rows = [
        ('D̄', 'reconstruction error', 'mean over steps and channels of (x − g(f(x)))² / σ²_x,  normal + abnormal',
         'f, g', '#0f8a6a'),
        ('L0', 'q0 negative log-likelihood', '−log q0(z_(t+1) | z_≤t) on NORMAL flights, per number per step',
         'q0 (and f via H)', '#2f6db5'),
        ('L1', 'q1 negative log-likelihood', '−log q1(z_(t+1) | z_≤t) on ABNORMAL flights, per number per step',
         'q1 (and f via H)', '#2f6db5'),
        ('H', 'modellability', 'H = L0 + L1:  codes that are easy to predict', 'f (weight η = 0.1)', '#2f6db5'),
        ('J', 'detection drift', 'mean over abnormal flights, t > t0 = 40 steps, of log q1(z) − log q0(z)\n'
         '(q held fixed: stop-gradient). This is the mean CUSUM increment after a change.', '—  (measured)',
         '#c2410c'),
        ('T', 'task loss', 'T = −log J, with J replaced by an EMA (β = 0.1) in the denominator for stability',
         'f (weight λ)', '#c2410c'),
        ('L', 'combined loss', 'L = (1 − λ) D̄ + λ T + η H', 'f, g', INK)]
    for i, (sym, nm, what, who, c) in enumerate(rows):
        y = 0.86 - i * 0.115
        box(ax, 0.0, y - 0.035, 0.05, 0.08, sym, c, fs=12, tc='white', bold=True)
        ax.text(0.065, y + 0.005, nm, fontsize=9.5, fontweight='bold', va='center')
        ax.text(0.27, y + 0.005, what, fontsize=8.8, va='center')
        ax.text(0.86, y + 0.005, who, fontsize=8.8, va='center', color=MUTED)
    ax.text(0.86, 0.94, 'trains', fontsize=9, fontweight='bold', color=MUTED)
    ax.text(0.0, 0.02, 'q0 and q1 are always fitted by maximum likelihood (minimise L0, L1) in their own inner steps; '
            'the encoder never gets to change q to improve J.  Validation versions of every term (val_D, val_L0, '
            'val_L1, val_H, val_J, val_T, val_loss) are computed on the full validation sets after every epoch.',
            fontsize=8.5, color=MUTED, wrap=True)
    save(fig, 'A4_loss_terms.png')


# ------------------------------------------------------------------ A5 schedule
def a5_schedule():
    fig, ax = plt.subplots(figsize=(15, 4.6))
    st = [('stage 0  reconstruct', 10, '#dbe7f7', 'train f, g on D̄ only\n(λ = 0, η = 0)'),
          ('stage 1  fit q0, q1', 10, '#e3f1ec', 'f, g frozen\nq0, q1 by max. likelihood'),
          ('stage 2  alternate', 15, '#fde7d4', 'per iteration: 3 q-steps, then 1 f,g-step on L\n'
           'λ ramps 0 → λ over the first 5 epochs'),
          ('stage 3  refit q0, q1', 10, '#ebe2f5', 'f, g frozen\nq0, q1 refit, lr cosine → 0')]
    x = 0
    for nm, n, c, txt in st:
        ax.barh(0, n, left=x, color=c, edgecolor='white', height=0.6)
        ax.text(x + n / 2, 0.12, nm, ha='center', fontsize=9.5, fontweight='bold')
        ax.text(x + n / 2, -0.1, txt, ha='center', va='center', fontsize=8.3)
        ax.text(x + n / 2, -0.42, f'{n} epochs', ha='center', fontsize=8.5, color=MUTED)
        x += n
    lam = np.concatenate([np.zeros(20), np.minimum(np.arange(1, 16) / 5, 1), np.ones(10)])
    ax2 = ax.twinx()
    ax2.step(np.arange(45) + 0.5, lam, where='mid', color='#c2410c', lw=2)
    ax2.set_ylim(-1.6, 1.4)
    ax2.set_yticks([0, 1], ['0', 'λ'])
    ax2.tick_params(colors='#c2410c')
    ax2.grid(False)
    ax.set_ylim(-0.6, 0.45)
    ax.set_yticks([])
    ax.set_xlim(0, 45)
    ax.set_xlabel('epoch (each = 70 iterations, batch 8 normal + 8 abnormal flights)')
    ax.set_title('A5  Training schedule (standard; long schedule = 20 / 20 / 30 / 20 epochs).  Adam, lr 1e-3, '
                 'gradient clip 1.0, η = 0.1, t0 = 40 steps.  Orange line: task weight used in that epoch.',
                 fontsize=10)
    save(fig, 'A5_schedule.png')


# ------------------------------------------------------------------ R1/R2 per anomaly type
SETTINGS = [
    ('raw Transformer', lambda: sel(pipeline='raw', q='tf')),
    ('raw GRU', lambda: sel(pipeline='raw', q='gru')),
    ('raw TCN', lambda: sel(pipeline='raw', q='tcn')),
    ('raw Linear AR', lambda: sel(pipeline='raw', q='lin')),
    ('TF k8 λ0', lambda: std_cae('tf', lam=0.0)),
    ('TF k8 λ0.1', lambda: std_cae('tf', lam=0.1)),
    ('TF k8 λ0.25', lambda: std_cae('tf', lam=0.25)),
    ('GRU k8 λ0.1', lambda: std_cae('gru', lam=0.1)),
    ('TCN k8 λ0.1', lambda: std_cae('tcn', lam=0.1)),
    ('TCN k8 λ0.25', lambda: std_cae('tcn', lam=0.25)),
    ('MoG TF k8 λ0', lambda: mog(lam=0.0)),
    ('MoG TF k8 λ0.1', lambda: mog(lam=0.1)),
    ('MoG TF k8 λ0.25', lambda: mog(lam=0.25)),
    ('MoG TF k64 λ0', lambda: mog(k=64, lam=0.0)),
    ('MoG TF k128 λ0', lambda: mog(k=128, lam=0.0)),
]


def r1_per_type():
    types = sorted(RUNS[0]['detect_per_type'])
    rows, DR, ADD, ns = [], [], [], []
    for nm, f in SETTINGS:
        rs = f()
        if not rs:
            continue
        rows.append(nm)
        ns.append(len(rs))
        DR.append([np.mean([m['detect_per_type'][t]['DR'] for m in rs]) * 100 for t in types])
        ADD.append([np.nanmean([np.nan if m['detect_per_type'][t]['ADD_median_s'] is None
                                else m['detect_per_type'][t]['ADD_median_s'] for m in rs]) for t in types])
    DR, ADD = np.array(DR), np.array(ADD)
    for name, A, cmap, lab, fmt in (('R1_detect_per_type.png', DR, 'Blues', 'detected %', '{:.0f}'),
                                    ('R2_delay_per_type.png', ADD, 'Oranges', 'median delay s', '{:.0f}')):
        fig, ax = plt.subplots(figsize=(15, 0.42 * len(rows) + 2.2))
        im = ax.imshow(A, cmap=cmap, aspect='auto', vmin=0, vmax=100 if 'detect' in name else np.nanmax(A))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                if not np.isnan(v):
                    dark = v > (60 if 'detect' in name else 0.6 * np.nanmax(A))
                    ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=7.5,
                            color='white' if dark else INK)
        ax.set_xticks(range(len(types)), [t.replace('_', ' ') for t in types], rotation=35, ha='right')
        ax.set_yticks(range(len(rows)), [f'{r}  (n={n})' for r, n in zip(rows, ns)])
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.02, label=lab)
        ttl = ('R1  Detected % per anomaly type (10 test flights per type, mean over seeds)' if 'detect' in name
               else 'R2  Median detection delay per anomaly type (s, detected flights only; blank = none detected)')
        ax.set_title(ttl)
        save(fig, name)
    return rows, types, DR


# ------------------------------------------------------------------ R3 drift curves
def r3_drift():
    groups = [('raw Transformer', sel(pipeline='raw', q='tf'), FAM['tf'][1], ':'),
              ('raw TCN', sel(pipeline='raw', q='tcn'), FAM['tcn'][1], ':'),
              ('TF k8 λ0', std_cae('tf', lam=0.0), FAM['tf'][1], '--'),
              ('TF k8 λ0.25', std_cae('tf', lam=0.25), FAM['tf'][1], '-'),
              ('TCN k8 λ0.25', std_cae('tcn', lam=0.25), FAM['tcn'][1], '-'),
              ('MoG TF k8 λ0', mog(lam=0.0), MOGC, '--'),
              ('MoG TF k8 λ0.1', mog(lam=0.1), MOGC, '-')]
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
    for nm, rs, c, ls in groups:
        if not rs:
            continue
        keys = sorted(rs[0]['drift_curve_val_change'], key=float)
        x = np.array([float(k_) for k_ in keys]) / HZ + 2.5
        # standardised: (increment - mean on normal) / sd on normal (val_normal, t > t0), so models whose
        # log-likelihoods live on very different scales (code scale, sigma floor, k) can share one axis
        Y = np.array([[(m['drift_curve_val_change'][k_] - m['varnorm']['mu0']) / m['varnorm']['sd0']
                       for k_ in keys] for m in rs])
        ax = axs[0] if nm.startswith('raw') else axs[1]
        ax.plot(x, Y.mean(0), color=c, ls=ls, lw=2, label=f'{nm} (n={len(rs)})')
    for ax, t in zip(axs, ('raw data (no compression)', 'compressed codes, k = 8')):
        ax.axvline(0, color=INK, lw=1)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel('time since the anomaly starts (s)')
        ax.set_ylabel('increment in normal-flight SDs: (s_t - mean) / sd')
        ax.set_yscale('symlog', linthresh=1)
        ax.set_title(f'R3  Post-change drift, validation change flights: {t}')
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'R3_drift_curves.png')


# ------------------------------------------------------------------ R4/R5 scatter of all runs
def kind_of(m):
    a = m['a']
    if a['pipeline'] == 'raw':
        return 'raw'
    return 'mog' if a['head'] == 'mog' else 'cae'


def r4_scatter():
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
    mk = {'raw': ('*', 'raw data'), 'cae': ('o', 'compressed, diag. Gaussian q'), 'mog': ('D', 'compressed, MoG q')}
    seen = set()
    for m in RUNS:
        a = m['a']
        kd = kind_of(m)
        c = MOGC if kd == 'mog' else FAM[a['q'] if kd == 'raw' else a['enc']][1]
        lab = None
        key = (kd, c)
        if key not in seen:
            seen.add(key)
            lab = f"{mk[kd][1]}: {'MoG TF' if kd == 'mog' else FAM[a['q'] if kd == 'raw' else a['enc']][0]}"
        filled = bool(m['test']['signs_ok'])
        axs[0].scatter(m['FA_test_normal'] * 100, m['detect']['DR'] * 100, marker=mk[kd][0], s=55 if kd != 'cae'
                       else 28, facecolor=c if filled else 'white', edgecolor=c, lw=1.1, label=lab, alpha=0.85)
        if kd != 'raw' and m.get('D_test_normal') is not None:
            axs[1].scatter(m['D_test_normal'], m['detect']['DR'] * 100, marker=mk[kd][0], s=45 if kd == 'mog' else 26,
                           facecolor=c if filled else 'white', edgecolor=c, lw=1.1, alpha=0.85)
    axs[0].axvline(5, color=MUTED, ls='--', lw=1)
    axs[0].text(5.2, 2, 'target 5 %', fontsize=8, color=MUTED)
    axs[0].set_xlabel('false alarm % on test normal flights')
    axs[0].set_ylabel('detected % on test change flights')
    axs[0].set_title(f'R4  Every run ({len(RUNS)}): detection vs false alarms  (hollow = sign check failed)')
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8, ncol=5, loc='lower center')
    axs[1].set_xlabel('reconstruction error D̄ on test normal flights')
    axs[1].set_ylabel('detected %')
    axs[1].set_title('R5  Reconstruction vs detection (compressed runs, all k and λ)')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, 'R4_R5_all_runs.png')


# ------------------------------------------------------------------ R6 sign check val vs test
def r6_signs():
    cae = [m for m in RUNS if 'val' in m and 'E0s' in m['val']]
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    dis = 0
    for m in cae:
        v, t = m['val'], m['test']
        agree = v['signs_ok'] == t['signs_ok']
        dis += not agree
        c = MOGC if m['a'].get('head') == 'mog' else (FAM[m['a']['q']][1] if m['a']['pipeline'] == 'raw'
                                                        else FAM[m['a']['enc']][1])
        axs[0].scatter(v['E0s'], t['E0s'], s=26, color=c if agree else 'white', edgecolor=c if agree else '#d62728',
                       lw=1.3)
    lim = [-5, 5]
    axs[0].plot(lim, lim, color=MUTED, lw=0.8)
    axs[0].axhline(0, color=INK, lw=0.6)
    axs[0].axvline(0, color=INK, lw=0.6)
    axs[0].set_xlim(lim)
    axs[0].set_ylim(lim)
    axs[0].set_xlabel('E0 on validation (mean increment on normal flights; must be < 0)')
    axs[0].set_ylabel('E0 on test')
    axs[0].set_title(f'R6a  Sign check, validation vs test: {dis} of {len(cae)} runs disagree (red ring)')
    ex = [('TCN k8 λ0.25 s0', std_cae('tcn', lam=0.25, seed=0)), ('MoG k8 λ0.1 s0', [m for m in mog(lam=0.1)
                                                                                      if m['a']['seed'] == 0]),
          ('TF k8 λ0.25 s0', std_cae('tf', lam=0.25, seed=0))]
    w = 0.13
    names = ['CE00', 'CE01', 'CE10', 'CE11']
    for i, (nm, rs) in enumerate(ex):
        if not rs:
            continue
        m = rs[0]
        for j, (sp, hatch) in enumerate((('val', ''), ('test', '//'))):
            axs[1].bar(np.arange(4) + (i * 2 + j - 2.5) * w, [m[sp][n] for n in names], w,
                       color=['#2f6db5', '#c2410c', MOGC][i], hatch=hatch, edgecolor='white',
                       label=f'{nm} {sp}')
    axs[1].set_xticks(range(4), ['CE00\nq0 on normal', 'CE01\nq1 on normal', 'CE10\nq0 on abnormal',
                                 'CE11\nq1 on abnormal'])
    axs[1].set_ylabel('cross-entropy (nats per step)')
    axs[1].set_title('R6b  The four cross-entropies (valid if CE01 > CE00 and CE10 > CE11)')
    axs[1].legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    save(fig, 'R6_sign_check.png')
    return dis, len(cae)


# ------------------------------------------------------------------ R7/R8 MoG training curves across λ and k
def curve(m, key):
    return np.array([e.get(key, np.nan) if e.get(key) is not None else np.nan for e in m['log']], float)


def stage_shade(ax, m):
    cols = {'s0': '#dbe7f7', 's1': '#e3f1ec', 's2': '#fde7d4', 's3': '#ebe2f5'}
    st = [e['stage'] for e in m['log']]
    i = 0
    while i < len(st):
        j = i
        while j < len(st) and st[j] == st[i]:
            j += 1
        ax.axvspan(i - 0.5, j - 0.5, color=cols.get(st[i], 'white'), alpha=0.5, lw=0)
        i = j


def r7_mog_lambda():
    keys = [('val_D', 'D̄ validation'), ('val_L0', 'L0 validation (q0 on normal)'), ('val_J', 'J validation'),
            ('val_loss', 'combined L validation'), ('z_std', 'code scale (std of z)'), ('pi_max', 'largest π')]
    cols = {0.0: '#9ca3af', 0.1: MOGC, 0.25: '#c2410c'}
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    shaded = False
    for lam, c in cols.items():
        for m in mog(lam=lam):
            for ax, (kk, t) in zip(axs.flat, keys):
                if not shaded:
                    stage_shade(ax, m)
                ax.plot(curve(m, kk), color=c, lw=1.4, ls='-' if m['a']['seed'] == 0 else '--',
                        label=f'λ = {lam:g}, seed {m["a"]["seed"]}')
                ax.set_title(t)
            shaded = True
    axs[1, 1].set_yscale('log')
    axs[0, 0].legend(frameon=False, fontsize=7)
    for ax in axs[1]:
        ax.set_xlabel('epoch (all stages)')
    fig.suptitle('R7  Supervisor spec (MoG q, linear encoder output, k = 8): validation curves for each λ  '
                 '(stages shaded: reconstruct / fit q / alternate / refit)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, 'R7_mog_curves_by_lambda.png')


def r8_mog_k():
    ks = sorted({m['a']['k'] for m in sel(head='mog', pipeline='cae', enc_out='linear', lam=0.0)})
    cm = plt.get_cmap('viridis')
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for i, k in enumerate(ks):
        for m in mog(k=k, lam=0.0)[:1]:
            c = cm(i / max(1, len(ks) - 1))
            for ax, kk, t in zip(axs, ('val_D', 'val_L0', 'z_std', 'val_J'),
                                 ('D̄ validation', 'L0 validation (per number)', 'code scale std(z)',
                                  'J validation')):
                ax.plot(curve(m, kk), color=c, lw=1.5, label=f'k = {k}')
                ax.set_title(t)
                ax.set_xlabel('epoch')
    axs[0].legend(frameon=False, fontsize=8)
    axs[2].set_yscale('log')
    fig.suptitle('R8  MoG head, λ = 0, seed 0: validation curves for each code size k', x=0.01, ha='left',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, 'R8_mog_curves_by_k.png')


# ------------------------------------------------------------------ R9 family training curves (old head)
def r9_family_curves():
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for fam in ('tf', 'gru', 'tcn'):
        rs = std_cae(fam, lam=0.25, seed=0)
        if not rs:
            continue
        m = rs[0]
        for ax, kk, t in zip(axs, ('D', 'L0', 'L1', 'J'), ('D̄ train', 'L0 train', 'L1 train', 'J train')):
            ax.plot(curve(m, kk), color=FAM[fam][1], lw=1.5, label=FAM[fam][0])
            ax.set_title(t)
            ax.set_xlabel('epoch')
    axs[0].legend(frameon=False)
    fig.suptitle('R9  Training curves of the three neural families (diag. Gaussian q, k = 8, λ = 0.25, seed 0)',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, 'R9_family_curves.png')


# ------------------------------------------------------------------ R10 λ x family heatmaps
def r10_heat():
    lams = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    rows = [('Transformer', lambda l: std_cae('tf', lam=l)), ('GRU', lambda l: std_cae('gru', lam=l)),
            ('TCN', lambda l: std_cae('tcn', lam=l)), ('Transformer + MoG (spec)', lambda l: mog(lam=l))]
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.4))
    for ax, (key, t, f, cmap) in zip(axs, (
            ('DR', 'detected %', lambda m: m['detect']['DR'] * 100, 'Blues'),
            ('ADD', 'median delay s', lambda m: m['detect']['ADD_median_s'], 'Oranges_r'),
            ('V', 'valid seeds %', lambda m: 100.0 * bool(m['test']['signs_ok']), 'Greens'))):
        A = np.full((len(rows), len(lams)), np.nan)
        N = np.zeros_like(A)
        for i, (_, g) in enumerate(rows):
            for j, l in enumerate(lams):
                mu, _, n = mean(g(l), f)
                A[i, j], N[i, j] = mu, n
        ax.imshow(A, cmap=cmap, aspect='auto')
        lo, hi = np.nanmin(A), np.nanmax(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if not np.isnan(A[i, j]):
                    fr = (A[i, j] - lo) / max(hi - lo, 1e-9)
                    dark = fr < 0.35 if cmap.endswith('_r') else fr > 0.6
                    ax.text(j, i, f'{A[i, j]:.0f}\n(n={int(N[i, j])})', ha='center', va='center', fontsize=7.5,
                            color='white' if dark else INK)
        ax.set_xticks(range(len(lams)), [f'λ={l:g}' for l in lams])
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.grid(False)
        ax.set_title(t)
    fig.suptitle('R10  Family × λ at k = 8 (mean over seeds; blank = not run)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, 'R10_family_lambda_heat.png')


# ------------------------------------------------------------------ R11 parameters and training time
def r11_cost():
    fams = ['tf', 'gru', 'tcn']
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.2))
    comp = {'encoder f': [nparam(models.Encoder(f, D_IN, 8)) for f in fams],
            'decoder g': [nparam(models.make_decoder('conv', D_IN, 8))] * 3,
            'q0 + q1 (diag.)': [2 * nparam(models.make_q('gauss', f, 8)) for f in fams],
            'q0 + q1 (MoG)': [2 * nparam(models.make_q('mog', f, 8)) for f in fams]}
    x = np.arange(3)
    for i, (nm, v) in enumerate(comp.items()):
        axs[0].bar(x + (i - 1.5) * 0.2, v, 0.2, label=nm, color=['#2f6db5', '#0f8a6a', '#b5862f', MOGC][i])
    axs[0].set_xticks(x, [FAM[f][0] for f in fams])
    axs[0].set_ylabel('parameters')
    axs[0].set_title('R11a  Parameters per component at k = 8')
    axs[0].legend(frameon=False, fontsize=8)
    groups = [(f'{FAM[f][0]} (laptop)', [m['train_s'] / 60 for m in std_cae(f, lam=0.25) if m['src'] == 'laptop'])
              for f in fams]
    groups += [('MoG TF k8 (server, shared GPU)', [m['train_s'] / 60 for m in sel(head='mog', k=8)]),
               ('MoG TF k64-128 (server)', [m['train_s'] / 60 for m in sel(head='mog') if m['a'].get('k', 0) >= 64])]
    groups = [(n, v) for n, v in groups if v]
    axs[1].bar(range(len(groups)), [np.mean(v) for n, v in groups], color=MUTED)
    for i, (n, v) in enumerate(groups):
        axs[1].text(i, np.mean(v), f'{np.mean(v):.0f} min\n(n={len(v)})', ha='center', va='bottom', fontsize=8)
    axs[1].set_xticks(range(len(groups)), [n for n, v in groups], rotation=12, ha='right', fontsize=8)
    axs[1].set_ylabel('wall-clock minutes per run')
    axs[1].set_title('R11b  Training time per run (45 epochs)')
    fig.tight_layout()
    save(fig, 'R11_cost.png')


# ------------------------------------------------------------------ R12 threshold and calibration
def r12_threshold():
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.2))
    for m in RUNS:
        a = m['a']
        if a['pipeline'] != 'cae':
            continue
        c = MOGC if a['head'] == 'mog' else FAM[a['enc']][1]
        axs[0].scatter(a['k'], m['h'], color=c, s=20, alpha=0.8)
        axs[1].scatter(m['varnorm']['FA_test_normal'] * 100, m['FA_test_normal'] * 100, color=c, s=20, alpha=0.8)
    axs[0].set_xscale('log', base=2)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('code size k')
    axs[0].set_ylabel('CUSUM threshold h (set on val_normal)')
    axs[0].set_title('R12a  The threshold grows with k: the increment is summed over k dimensions')
    axs[1].plot([0, 15], [0, 15], color=MUTED, lw=0.8)
    axs[1].axhline(5, color=MUTED, ls='--', lw=0.8)
    axs[1].axvline(5, color=MUTED, ls='--', lw=0.8)
    axs[1].set_xlabel('false alarm % with variance-normalised increments')
    axs[1].set_ylabel('false alarm % (main metric)')
    axs[1].set_title('R12b  False alarms on held-out normals for the two calibrations (target 5 %)')
    for f in ('tf', 'gru', 'tcn'):
        axs[0].scatter([], [], color=FAM[f][1], label=FAM[f][0])
    axs[0].scatter([], [], color=MOGC, label='MoG TF')
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, 'R12_threshold.png')


if __name__ == '__main__':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    print(len(RUNS), 'runs loaded')
    P = a1_pipeline()
    print('params k=8:', P)
    a2_mog()
    a3_families()
    a4_losses()
    a5_schedule()
    rows, types, DR = r1_per_type()
    for r, d in zip(rows, DR):
        print(f'{r:18s}', ' '.join(f'{v:3.0f}' for v in d))
    print('types', types)
    r3_drift()
    r4_scatter()
    print('sign disagree', r6_signs())
    r7_mog_lambda()
    r8_mog_k()
    r9_family_curves()
    r10_heat()
    r11_cost()
    r12_threshold()
