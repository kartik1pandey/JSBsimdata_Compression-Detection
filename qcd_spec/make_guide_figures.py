"""Explanatory figures for docs/GUIDE.md (dataset, pipeline, models, training schedule, CUSUM, calibration).
Data figures use the real qcd_pure dataset and the stored LLRs of a finished run (default raw_qtf_s0).
    python make_guide_figures.py [run_tag]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DATA = Path(r'C:\Users\karti\IISc_data\qcd_pure')
CKPT = Path(r'C:\Users\karti\IISc_data\qcd_spec_ckpt')
HERE = Path(__file__).resolve().parent
FIG = HERE / 'docs' / 'figs'
INK, MUTED, GRID = '#222222', '#6b6b6b', '#e2e2e2'
C_N, C_A, C_Z, C_Q = '#2f6db5', '#d0603a', '#3a9b6b', '#7a6bb8'          # normal, abnormal, latent, density
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#bbbbbb', 'axes.grid': True, 'grid.color': GRID,
                     'axes.spines.top': False, 'axes.spines.right': False, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
CH = json.loads((DATA / 'metadata.json').read_text())['channels']
T = 1200
t_s = np.arange(T) / 4


def mm(r):
    return np.load(DATA / f'{r}_x.npy', mmap_mode='r')


def meta(r):
    return dict(np.load(DATA / f'{r}_meta.npz'))


def box(ax, x, y, w, h, text, fc, ec=None, fs=9, tc='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc,
                                ec=ec or fc, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc)


def arrow(ax, x0, y0, x1, y1, c=MUTED, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=12, color=c, lw=1.4, ls=ls))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    return fig, ax


# ------------------------------------------------------------------ 1. three kinds of trajectory (schematic)
def fig_kinds():
    fig, ax = blank((9, 3.2))
    ax.set_ylim(0, 3.4)
    rows = [('Normal (D0)', 'normal for all 300 s', None), ('Pure abnormal (D1)', 'anomaly active from sample 1', 0.0),
            ('Change (test only)', None, 0.45)]
    for i, (name, note, cp) in enumerate(rows):
        y = 2.6 - i * 1.05
        ax.text(0.0, y + 0.22, name, fontsize=10, va='center', fontweight='bold')
        x0, x1 = 2.6, 9.8
        if cp is None:
            box(ax, x0, y, x1 - x0, 0.45, note, C_N)
        elif cp == 0.0:
            box(ax, x0, y, x1 - x0, 0.45, note, C_A)
        else:
            xm = x0 + cp * (x1 - x0)
            box(ax, x0, y, xm - x0, 0.45, 'normal', C_N)
            box(ax, xm, y, x1 - xm, 0.45, 'abnormal from ν to the end', C_A)
            ax.annotate('change point ν\n(sample 100–800)', (xm, y), (xm, y - 0.45), ha='center', fontsize=8,
                        color=MUTED, arrowprops=dict(arrowstyle='-', color=MUTED))
    ax.text(2.6, 3.25, '0 s', fontsize=8, color=MUTED)
    ax.text(9.6, 3.25, '300 s', fontsize=8, color=MUTED)
    ax.text(5.6, 3.25, 'T = 1200 samples at 4 Hz (flight time 720–1020 s)', fontsize=8, color=MUTED, ha='center')
    fig.tight_layout()
    fig.savefig(FIG / '01_trajectory_kinds.png', dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 2. real examples: same channel, 3 kinds
def fig_examples():
    mc = meta('test_change')
    ty = mc['anomaly_type'].astype(str)
    xc = mm('test_change')
    # the change trajectory with the clearest response: largest standardised post/pre shift in any channel
    best = None
    for i in np.where(np.isin(ty, ['surface_bias', 'throttle_thrust_loss', 'sensor_bias', 'surface_runaway']))[0]:
        cp = int(mc['change_point'][i])
        x = np.asarray(xc[i])
        sc = np.abs(x[cp + 40:].mean(0) - x[:cp].mean(0)) / (x[:cp].std(0) + 0.05)
        c = int(sc.argmax())
        if best is None or sc[c] > best[0]:
            best = (sc[c], i, c, cp)
    _, i, c, cp = best
    ma = meta('test_abnormal')
    j = int(np.where(ma['anomaly_type'].astype(str) == ty[i])[0][0])
    fig, axs = plt.subplots(3, 1, figsize=(8.5, 6.2), sharex=True)
    for ax, x, lab, col, title in ((axs[0], mm('test_normal')[0, :, c], 'normal', C_N, 'Normal trajectory (test_normal)'),
                                   (axs[1], mm('test_abnormal')[j, :, c], 'abnormal', C_A,
                                    f'Pure abnormal trajectory ({ty[i]}, test_abnormal)'),
                                   (axs[2], xc[i, :, c], 'change', C_A, f'Change trajectory ({ty[i]}, test_change)')):
        x = np.asarray(x)
        if lab == 'change':
            ax.plot(t_s[:cp], x[:cp], color=C_N, lw=1.4)
            ax.plot(t_s[cp:], x[cp:], color=C_A, lw=1.4)
            ax.axvline(cp / 4, color=INK, lw=1, ls='--')
            ax.text(cp / 4 + 2, ax.get_ylim()[1], ' ν', va='top', fontsize=9)
        else:
            ax.plot(t_s, x, color=col, lw=1.4)
        ax.set_title(title, fontsize=9, loc='left')
        ax.set_ylabel('z-score')
    axs[2].set_xlabel('seconds into the trajectory')
    fig.suptitle(f'Channel {CH[c]} in three kinds of trajectory (real data)', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / '02_real_examples.png', dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 3. the three pipelines
def fig_pipelines():
    fig, ax = blank((9.5, 4.6))
    ax.set_ylim(0, 4.8)
    y = [3.9, 2.4, 0.9]
    names = ['(i) raw', '(ii) unaware AE  (λ = 0)', '(iii) task-aware  (λ > 0)']
    for yy, nm in zip(y, names):
        ax.text(0.0, yy + 0.25, nm, fontsize=9, fontweight='bold', va='center')
    # raw
    box(ax, 2.2, y[0], 1.3, 0.5, 'x_t\n297 channels', C_N)
    box(ax, 5.2, y[0], 1.9, 0.5, 'q0, q1 on x\n(next-step density)', C_Q)
    box(ax, 8.0, y[0], 1.6, 0.5, 'LLR s_t → CUSUM\nalarm if W_t ≥ h', '#444444')
    arrow(ax, 3.5, y[0] + 0.25, 5.2, y[0] + 0.25)
    arrow(ax, 7.1, y[0] + 0.25, 8.0, y[0] + 0.25)
    for yy, extra in ((y[1], 'trained on D̄ (+ηH̄)'), (y[2], 'trained on (1−λ)D̄ + λ(−log Ĵ) + ηH̄')):
        box(ax, 2.2, yy, 1.3, 0.5, 'x_t\n297 channels', C_N)
        box(ax, 3.9, yy, 1.0, 0.5, 'encoder f\n(causal)', C_Z)
        box(ax, 5.2, yy, 1.9, 0.5, 'q0, q1 on z\n(k = 8 dims)', C_Q)
        box(ax, 8.0, yy, 1.6, 0.5, 'LLR s_t → CUSUM\nalarm if W_t ≥ h', '#444444')
        arrow(ax, 3.5, yy + 0.25, 3.9, yy + 0.25)
        arrow(ax, 4.9, yy + 0.25, 5.2, yy + 0.25)
        arrow(ax, 7.1, yy + 0.25, 8.0, yy + 0.25)
        box(ax, 3.9, yy - 0.62, 1.0, 0.38, 'decoder g', '#9aa5b1', fs=8)
        arrow(ax, 4.4, yy, 4.4, yy - 0.24, ls='--')
        ax.text(5.05, yy - 0.43, 'x̂ (offline only, gives D̄)', fontsize=7.5, color=MUTED, va='center')
        ax.text(3.9, yy + 0.62, 'encoder ' + extra, fontsize=8, color=C_Z)
    fig.tight_layout()
    fig.savefig(FIG / '03_pipelines.png', dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 4. model families (context each sees)
def fig_families():
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.9))
    Tn, m = 48, 12
    i = np.arange(Tn)
    d = i[:, None] - i[None, :]
    axs[0].imshow((d >= 0) & (d < m), cmap='Blues', interpolation='nearest')
    axs[0].set_title('Causal transformer\nattends to last m steps', fontsize=9)
    axs[0].set_xlabel('context position')
    axs[0].set_ylabel('predicted position')
    axs[1].imshow(d >= 0, cmap='Oranges', interpolation='nearest')
    axs[1].set_title('GRU\nwhole history (in a state)', fontsize=9)
    axs[1].set_xlabel('context position')
    for ax in axs[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    ax = axs[2]
    for L_, q in enumerate((1, 2, 4, 8)):
        ax.scatter(np.arange(16), np.full(16, L_), s=10, color='#cccccc')
    tgt = 15
    pts = {tgt}
    for L_, q in reversed(list(enumerate((1, 2, 4, 8)))):
        new = set()
        for p in pts:
            for pp in (p, p - q):
                if pp >= 0:
                    ax.plot([p, pp], [L_ + 1, L_], color=C_Z, lw=1)
                    new.add(pp)
        pts = new
    ax.scatter([tgt], [4], s=30, color=C_Z)
    ax.set_title('TCN (dilations 1,2,4,…)\nreceptive field = 2^layers', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax = axs[3]
    ax.scatter(np.arange(16), np.zeros(16), s=10, color='#cccccc')
    for p in range(11, 15):
        ax.plot([p, 15], [0, 1], color=C_Q, lw=1)
    ax.scatter([15], [1], s=30, color=C_Q)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('Linear AR(4)\nlast 4 steps, linear', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / '04_model_families.png', dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 5. training schedule
def fig_schedule():
    e0, e1, e2, e3, warm = 10, 10, 15, 10, 5
    ep = np.arange(e0 + e1 + e2 + e3)
    lam_target = 0.5
    lam = np.zeros(len(ep))
    for k in range(e2):
        lam[e0 + e1 + k] = lam_target * min(1, (k + 1) / warm)
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax = axs[0]
    spans = [(0, e0, 'Stage 0\nwarm-start f, g\nmin D̄', C_Z), (e0, e0 + e1, 'Stage 1\nf frozen\nfit q0 on D0, q1 on D1', C_Q),
             (e0 + e1, e0 + e1 + e2, 'Stage 2: alternate\n3 steps (q0,q1) : 1 step (f,g)\nλ ramps up', '#444444'),
             (e0 + e1 + e2, len(ep), 'Stage 3\nf frozen\nrefit q0, q1, lr → 0', C_Q)]
    for a_, b_, txt, c in spans:
        ax.axvspan(a_, b_, color=c, alpha=0.15)
        ax.text((a_ + b_) / 2, 0.5, txt, ha='center', va='center', fontsize=8.5)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.set_title('Compressed pipeline training schedule (raw pipeline: stage 1 only, 30 epochs, lr → 0)', fontsize=9)
    axs[1].step(ep, lam, where='post', color=INK, lw=2)
    axs[1].set_ylabel('λ (example\ntarget 0.5)')
    axs[1].set_xlabel('epoch (70 iterations of 8 normal + 8 abnormal trajectories)')
    fig.tight_layout()
    fig.savefig(FIG / '05_training_schedule.png', dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 6/7. CUSUM on a real trajectory + calibration
def fig_cusum(tag):
    z = np.load(CKPT / f'{tag}_llr.npz')
    m = json.loads((HERE / 'results' / f'{tag}.json').read_text())
    h = m['h']
    s_all = z['test_change']
    nu = meta('test_change')['change_point'].astype(int)
    stop = z['stop']
    det = np.where((stop >= nu) & (stop >= 0))[0]
    i = int(det[np.argsort(stop[det] - nu[det])[len(det) // 2]]) if len(det) else 0
    s = s_all[i]
    W = np.zeros_like(s)
    w = 0.0
    for t in range(41, len(s)):
        w = max(0.0, w + s[t])
        W[t] = w
    tt = (np.arange(len(s)) + 1) / 4
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True)
    sym = lambda v: np.sign(v) * np.log10(1 + np.abs(v))   # noqa: E731
    axs[0].plot(tt, sym(s), color=MUTED, lw=0.8)
    axs[0].axhline(0, color=INK, lw=0.8)
    axs[0].set_ylabel('s_t = log q1 − log q0\n(sign·log10(1+|s|))')
    axs[1].plot(tt, W, color=C_A, lw=1.8, label='CUSUM W_t')
    axs[1].axhline(h, color=INK, ls='--', lw=1, label=f'threshold h = {h:.3g}')
    for ax in axs:
        ax.axvline(nu[i] / 4, color=C_N, lw=1.2)
    if stop[i] >= 0:
        axs[1].axvline(stop[i] / 4, color=C_A, lw=1.2, ls=':')
        axs[1].text(stop[i] / 4, h, f'  alarm, delay {(stop[i] - nu[i]) / 4:.1f} s', fontsize=8, va='bottom')
    axs[1].text(nu[i] / 4, axs[1].get_ylim()[1] * 0.9, ' change ν', fontsize=8, color=C_N)
    axs[1].set_ylabel('W_t (nats)')
    axs[1].set_xlabel('seconds')
    axs[1].legend(frameon=False, fontsize=8, loc='upper left')
    axs[0].set_title(f'CUSUM on one test_change trajectory ({tag}, median-delay detection)', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / '06_cusum_example.png', dpi=150)
    plt.close(fig)
    # calibration: max W per val_normal trajectory, order-statistic threshold
    sv = z['val_normal']
    Mx = []
    for r in sv:
        w, mx = 0.0, 0.0
        for t in range(41, len(r)):
            w = max(0.0, w + r[t])
            mx = max(mx, w)
        Mx.append(mx)
    Mx = np.sort(np.array(Mx))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(np.arange(1, len(Mx) + 1), Mx, 'o', ms=3.5, color=C_N, label='M_i = max_t W_t, sorted (val_normal)')
    r_ = m['calibration']['r']
    ax.plot([r_], [Mx[r_ - 1]], 'o', ms=8, color=C_A, label=f'h = M_(r), r = ⌈(N+1)(1−α)⌉ = {r_}')
    ax.set_yscale('symlog')
    ax.set_xlabel('rank among N = %d normal validation trajectories' % len(Mx))
    ax.set_ylabel('max CUSUM (nats)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f'Threshold calibration: about 5 % of normal trajectories exceed h ({tag})', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / '07_calibration.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    FIG.mkdir(parents=True, exist_ok=True)
    tag = sys.argv[1] if len(sys.argv) > 1 else 'raw_qtf_s0'
    fig_kinds()
    fig_examples()
    fig_pipelines()
    fig_families()
    fig_schedule()
    fig_cusum(tag)
    print('figures in', FIG)
