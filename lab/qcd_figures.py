"""Figures that explain the QCD dataset: python -m lab.qcd_figures [--data ./qcd_data]

Writes <data>/figures/01_timeline.png ... 06_counts.png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import qcd_dataset as q  # noqa: E402
from .aircraft import LOG_HZ  # noqa: E402

HERE = Path(__file__).parent
NORMAL, ABN, WIN = '#3b6fb6', '#d1603d', '#e9e4da'
INK, MUTED = '#222222', '#6b6b6b'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#bbbbbb', 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': '#eeeeee', 'grid.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.4,
                     'figure.dpi': 110, 'savefig.bbox': 'tight'})


def save(fig, out, name):
    fig.savefig(out / name)
    plt.close(fig)
    print('  ', out / name)


def timeline(meta, out):
    W0 = meta['window_starts_s'][0]
    T_s = meta['window']['seconds']
    cw = meta['change_window_start_s']
    cp = [cw + c / LOG_HZ for c in meta['window']['change_point_range']]
    fig, ax = plt.subplots(figsize=(11, 3.0))
    lanes = [('Normal flight', None), ('Change flight', cp)]
    for i, (name, rng) in enumerate(lanes):
        y = 2 - i
        ax.barh(y, meta['flight_s'], left=0, height=0.5, color='#f4f4f4', edgecolor='#cccccc')
        if rng:
            ax.barh(y, meta['flight_s'] - rng[0], left=rng[0], height=0.5, color=ABN, alpha=0.18)
            ax.barh(y, rng[1] - rng[0], left=rng[0], height=0.5, color=ABN, alpha=0.45,
                    hatch='///', edgecolor='white')
            ax.text(rng[1] + 60, y, f'<- anomaly starts somewhere in {rng[0]:g}-{rng[1]:g} s\n    and lasts to the end',
                    ha='left', va='center', fontsize=8.5, color=INK)
        ws = cw if i == 1 else W0
        ax.barh(y, T_s, left=ws, height=0.5, fill=False, edgecolor=INK, linewidth=2)
        ax.text(-20, y, name, ha='right', va='center', fontsize=10, color=INK)
    ax.annotate('', xy=(W0, 0.45), xytext=(W0 + T_s, 0.45), arrowprops=dict(arrowstyle='<->', color=INK))
    ax.text(W0 + T_s / 2, 0.3, f'the sample: {W0:g}-{W0 + T_s:g} s = {T_s:g} s = T {meta["window"]["T"]} steps '
            f'(4 per second)', ha='center', va='top', fontsize=9.5, color=INK)
    ax.set_xlim(0, meta['flight_s'])
    ax.set_ylim(-0.2, 2.6)
    ax.set_yticks([])
    ax.set_xlabel('flight time (s)')
    ax.grid(False)
    ax.set_title('Every sample is cut from the same 120 s of its flight (black box)', loc='left', color=INK)
    save(fig, out, '01_timeline.png')


def sample_matrix(d, meta, out):
    z = np.load(d / 'test_normal.npz')
    x = z['x'][0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1]))
    im = a1.imshow(x.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, origin='lower',
                   extent=[0, x.shape[0] / LOG_HZ, 0, x.shape[1]])
    a1.set_xlabel('time inside the sample (s)   [480 steps, one every 0.25 s]')
    a1.set_ylabel(f'channel index (0..{x.shape[1] - 1})')
    a1.set_title(f'One sample = a {x.shape[0]} x {x.shape[1]} matrix (z-score, colour)', loc='left', color=INK)
    a1.grid(False)
    fig.colorbar(im, ax=a1, label='z (standard deviations from normal mean)', fraction=0.03)
    names = [c['name'] for c in meta['channels']]
    pick = ['position/h-sl-ft', 'velocities/vc-kts', 'attitude/theta-deg', 'propulsion/engine[0]/n1']
    for k, nm in enumerate(pick):
        j = names.index(nm)
        a2.plot(np.arange(x.shape[0]) / LOG_HZ, x[:, j] + 4 * (len(pick) - 1 - k), color=NORMAL)
        a2.text(121, 4 * (len(pick) - 1 - k), nm.split('/')[-1], va='center', fontsize=8.5, color=INK)
    a2.set_yticks([])
    a2.set_xlabel('time (s)')
    a2.set_title('4 of its rows (channels) as lines', loc='left', color=INK)
    save(fig, out, '02_sample_matrix.png')


def full_flight(d, meta, out):
    from .scenario import draw_spec
    from .simulate import add_sensor_noise, fly
    with open(d / 'flights.csv', encoding='utf-8') as f:
        fl = list(csv.DictReader(f))
    tasks = {t['id']: t for t in q.plan()}
    nrm = next(r for r in fl if r['role'] == 'test_normal' and not r['hard_normal'])
    ab = max((r for r in fl if r['role'] == 'test_change' and r['anomaly_type'] == 'throttle_thrust_loss'
              and not r['hard_normal']), key=lambda r: float(r['severity']))
    runs = []
    for r in (nrm, ab):
        t = tasks[r['flight_id']]
        spec = draw_spec(t['id'], t['seed'], t['hard'], duration=q.FLIGHT_S)
        a = q.draw_anomaly(t, int(r['attempts']) - 1) if t['typ'] else None
        runs.append((r, add_sensor_noise(fly(spec, a), spec.seed)))
    e = int(ab['engine']) - 1
    chans = [('position/h-sl-ft', 'altitude (ft)'), ('velocities/vc-kts', 'airspeed CAS (kt)'),
             (f'propulsion/engine[{e}]/thrust-lbs', f'engine {e + 1} thrust (lb)  <- faulty engine'),
             ('propulsion/engine[1]/thrust-lbs' if e != 1 else 'propulsion/engine[0]/thrust-lbs',
              'a healthy engine thrust (lb)')]
    fig, axs = plt.subplots(len(chans), 1, figsize=(11, 8), sharex=True)
    W0, T_s = meta['window_starts_s'][0], meta['window']['seconds']
    for ax, (c, lab) in zip(axs, chans):
        ax.axvspan(W0, W0 + T_s, color=WIN, zorder=0)
        for (r, df), col, nm in zip(runs, (NORMAL, ABN), ('normal', 'change')):
            ax.plot(df['t'], df[c], color=col, label=f"{nm} flight {r['flight_id']}")
        ax.axvline(float(ab['onset_s']), color=ABN, ls='--', lw=1)
        ax.set_ylabel(lab, fontsize=8.5, rotation=0, ha='right', va='center')
    axs[0].legend(loc='upper left', frameon=False, ncol=2, fontsize=9)
    axs[0].text(float(ab['onset_s']) + 8, axs[0].get_ylim()[1], f"thrust loss starts {float(ab['onset_s']):.0f} s",
                color=ABN, fontsize=8.5, va='top')
    axs[0].text(W0 + T_s / 2, axs[0].get_ylim()[0], 'sample', ha='center', va='bottom', fontsize=9, color=INK)
    axs[-1].set_xlabel('flight time (s)')
    fig.suptitle(f"Whole 1500 s flights (raw units). Only the shaded 120 s goes into the dataset.\n"
                 f"Change flight: throttle_thrust_loss on engine {e + 1}, severity {float(ab['severity']):.2f}, "
                 f"{ab['profile']} profile", x=0.07, ha='left', color=INK)
    save(fig, out, '03_full_flight.png')


def change_sample(d, meta, out):
    z = np.load(d / 'test_change.npz')
    names = [c['name'] for c in meta['channels']]
    i = int(np.where(z['anomaly_type'] == 'throttle_lever_bias')[0][0])
    x, lab, cp = z['x'][i], z['label'][i], int(z['change_point'][i])
    dev = np.abs(x[cp:].mean(0) - x[:cp].mean(0))
    top = np.argsort(-dev)[:4]
    t = np.arange(len(x)) / LOG_HZ
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[4, 0.7]))
    for k, j in enumerate(top):
        a1.plot(t, x[:, j] + 0 * k, lw=1.2, label=names[j])
    a1.axvline(cp / LOG_HZ, color=ABN, ls='--')
    a1.text(cp / LOG_HZ + 1, a1.get_ylim()[1], f'change point: step {cp} = {cp / LOG_HZ:g} s\n'
            f'(flight time {meta["change_window_start_s"] + cp / LOG_HZ:g} s)', color=ABN, va='top', fontsize=9)
    a1.set_ylabel('z')
    a1.legend(frameon=False, fontsize=8, loc='lower left')
    a1.set_title(f"A change sample ({z['flight_id'][i]}, throttle_lever_bias): normal first, then the anomaly. "
                 'The 4 channels that change most', loc='left', color=INK, fontsize=10)
    a2.fill_between(t, lab, step='post', color=ABN, alpha=0.6)
    a2.set_yticks([0, 1])
    a2.set_ylabel('label')
    a2.set_xlabel('time inside the sample (s)')
    save(fig, out, '04_change_sample.png')


def target_channel(var):
    """The channel an anomaly acts on directly."""
    if var.startswith('engine'):
        return f'propulsion/engine[{int(var.split()[1]) - 1}]/thrust-lbs'
    surf = {'elevator': 'fcs/elevator-pos-rad', 'aileron': 'fcs/left-aileron-pos-rad', 'rudder': 'fcs/rudder-pos-rad'}
    if var in surf:
        return surf[var]
    return 'sensor/HDG:sin' if var == 'HDG' else f'sensor/{var}'


def types_grid(d, meta, out):
    zn, za = np.load(d / 'test_normal.npz'), np.load(d / 'test_change.npz')
    names = [c['name'] for c in meta['channels']]
    xn = zn['x']
    lo, hi = np.percentile(xn, 2.5, axis=0), np.percentile(xn, 97.5, axis=0)
    types = list(dict.fromkeys(za['anomaly_type'].tolist()))
    fig, axs = plt.subplots(4, 4, figsize=(15, 11), sharex=True)
    t = np.arange(xn.shape[1]) / LOG_HZ
    for ax, ty in zip(axs.flat, types):
        idx = np.where(za['anomaly_type'] == ty)[0]
        ch = np.array([names.index(target_channel(v)) for v in za['variable'][idx]])
        dev = np.array([np.abs(za['x'][i][za['change_point'][i]:, j] - xn[:, za['change_point'][i]:, j].mean(0)).mean()
                        for i, j in zip(idx, ch)])
        k = np.argsort(dev)[len(dev) // 2]                          # the median sample: typical, not extreme
        i, j = idx[k], ch[k]
        ax.fill_between(t, lo[:, j], hi[:, j], color=NORMAL, alpha=0.2, lw=0, label='normal (95 % of samples)')
        ax.plot(t, za['x'][i][:, j], color=ABN, lw=1.1, label='change sample')
        ax.axvline(za['change_point'][i] / LOG_HZ, color=INK, ls='--', lw=0.9)
        ax.set_title(f'{ty}\n{names[j]}', fontsize=8.5, loc='left', color=INK)
        ax.text(0.98, 0.04, f"sev {za['severity'][i]:.2f}, {za['variable'][i]}", transform=ax.transAxes,
                ha='right', fontsize=7.5, color=MUTED)
    for ax in axs.flat[len(types):]:
        ax.axis('off')
    axs.flat[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    for ax in axs[-1]:
        ax.set_xlabel('time in sample (s)')
    fig.suptitle('Each anomaly type: a typical (median) test change sample, on the channel the fault acts on, '
                 'against the range of normal samples (z units). Dashed line = change point.', x=0.07, ha='left',
                 color=INK)
    fig.tight_layout()
    save(fig, out, '05_types_grid.png')


def counts(meta, out):
    c = meta['counts']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1, 2.2]))
    splits = ['train', 'val', 'test']
    xx = np.arange(3)
    for off, kind, col in ((-0.2, 'normal', NORMAL), (0.2, 'change', ABN)):
        v = [c[f'{s}_{kind}']['windows'] for s in splits]
        bars = a1.bar(xx + off, v, 0.38, color=col, label=kind)
        for b in bars:
            a1.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{b.get_height():.0f}', ha='center',
                    va='bottom', fontsize=8.5, color=INK)
    a1.set_xticks(range(3), splits)
    a1.set_ylabel('samples')
    a1.legend(frameon=False, fontsize=8.5)
    a1.set_title('Samples per split (1 sample = 1 flight)', loc='left', color=INK)
    types = list(c['train_change']['per_type'])
    bottom = np.zeros(len(types))
    for role, col in (('train_change', '#d1603d'), ('val_change', '#e39a7f'), ('test_change', '#f0c7b6')):
        v = np.array([c[role]['per_type'].get(ty, 0) for ty in types])
        a2.barh(types, v, left=bottom, color=col, label=role.replace('_change', ''), edgecolor='white')
        bottom += v
    a2.invert_yaxis()
    a2.tick_params(axis='y', labelsize=8.5)
    a2.legend(frameon=False, fontsize=8.5, ncol=3, loc='lower right')
    a2.set_xlabel('samples')
    a2.set_title('Change samples per anomaly type (equal for all 14)', loc='left', color=INK)
    save(fig, out, '06_counts.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(HERE.parent / 'qcd_data'))
    ap.add_argument('--skip-flight', action='store_true', help='skip 03 (re-flies two flights)')
    d = Path(ap.parse_args().data)
    a = ap.parse_args()
    out = d / 'figures'
    out.mkdir(exist_ok=True)
    meta = json.loads((d / 'metadata.json').read_text(encoding='utf-8'))
    timeline(meta, out)
    sample_matrix(d, meta, out)
    if not a.skip_flight:
        full_flight(d, meta, out)
    change_sample(d, meta, out)
    types_grid(d, meta, out)
    counts(meta, out)


if __name__ == '__main__':
    main()
