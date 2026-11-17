r"""Figures for the Cruise Lab presentation, drawn from the demo data in ../data.

    ..\..\.venv\Scripts\python make_figures.py      ->  figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent / 'data'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)

FLIGHT, TWIN, TRUTH, OTHER = '#2a78d6', '#eb6834', '#6250d6', '#1baf7a'
INK, MUTED, GRID = '#1f2328', '#6b7280', '#e5e7eb'
plt.rcParams.update({
    'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'figure.dpi': 160, 'savefig.bbox': 'tight', 'lines.linewidth': 1.6,
})


def load(fid, cols):
    meta = json.loads((DATA / 'meta' / f'{fid}.json').read_text(encoding='utf-8'))
    df = pd.read_parquet(DATA / 'flights' / f'{fid}.parquet', columns=['t'] + cols)
    tw = None
    p = DATA / 'flights' / f'{fid}_twin.parquet'
    if p.exists():
        tw = pd.read_parquet(p, columns=['t'] + cols)
    return meta, df, tw


def shade(ax, a):
    if a is None:
        return
    spans = a.get('bursts') or [[a['onset_s'], a['end_s']]]
    for x, y in spans:
        ax.axvspan(x / 60, y / 60, color=TWIN, alpha=0.10, lw=0)
    for x in (a['onset_s'], a['end_s']):
        ax.axvline(x / 60, color=TWIN, lw=0.9, ls=':')


def finish(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print('wrote', name)


def m(t):
    return np.asarray(t) / 60.0


# 1. pitot blockage: reading vs truth vs twin
meta, df, tw = load('O105', ['sensor/CAS', 'velocities/vc-kts', 'propulsion/engine[0]/n1'])
a = meta['anomaly']
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
shade(ax[0], a)
ax[0].plot(m(tw.t), tw['velocities/vc-kts'], color=TWIN, label='normal twin')
ax[0].plot(m(df.t), df['velocities/vc-kts'], color=TRUTH, ls='--', label='true CAS')
ax[0].plot(m(df.t), df['sensor/CAS'], color=FLIGHT, lw=1.1, label='CAS reading (autopilot input)')
ax[0].set(xlabel='time [min]', ylabel='CAS [kt]', title='Reading stays on target, true speed falls')
ax[0].legend(loc='lower left', fontsize=9)
shade(ax[1], a)
ax[1].plot(m(tw.t), tw['propulsion/engine[0]/n1'], color=TWIN, label='normal twin')
ax[1].plot(m(df.t), df['propulsion/engine[0]/n1'], color=FLIGHT, lw=1.1, label='this flight')
ax[1].set(xlabel='time [min]', ylabel='N1 engine 1 [%]', title='Autothrottle pulls power back')
ax[1].legend(loc='lower left', fontsize=9)
finish(fig, 'pitot_O105.png')

# 2. altitude bias: autopilot holds the WRONG reading
meta, df, tw = load('O100', ['sensor/ALT', 'position/h-sl-ft', 'sel/alt-ft'])
a = meta['anomaly']
fig, ax = plt.subplots(figsize=(6.2, 3.4))
shade(ax, a)
ax.plot(m(df.t), df['sel/alt-ft'], color=MUTED, lw=1, ls='-.', label='selected altitude')
ax.plot(m(tw.t), tw['position/h-sl-ft'], color=TWIN, label='twin true altitude')
ax.plot(m(df.t), df['position/h-sl-ft'], color=TRUTH, ls='--', label='true altitude')
ax.plot(m(df.t), df['sensor/ALT'], color=FLIGHT, lw=1.0, label='ALT reading')
ax.set(xlabel='time [min]', ylabel='altitude [ft]',
       title=f"ALT bias {a['sign'] * a['magnitude']:+.0f} ft: aircraft settles off target")
ax.legend(loc='lower left', fontsize=8.5, ncol=2)
finish(fig, 'altbias_O100.png')

# 3. elevator stuck: command vs applied vs altitude
meta, df, tw = load('C101', ['ap/elevator-cmd', 'fcs/elevator-cmd-norm', 'position/h-sl-ft'])
a = meta['anomaly']
lo, hi = (a['onset_s'] - 120) / 60, (a['end_s'] + 240) / 60
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for x in ax:
    shade(x, a)
    x.set_xlim(lo, hi)
ax[0].plot(m(df.t), df['ap/elevator-cmd'], color=FLIGHT, label='autopilot command')
ax[0].plot(m(df.t), df['fcs/elevator-cmd-norm'], color=TRUTH, ls='--', label='applied to surface')
ax[0].set(xlabel='time [min]', ylabel='elevator [-1..1]', title='Command moves, surface frozen')
ax[0].legend(fontsize=9)
ax[1].plot(m(tw.t), tw['position/h-sl-ft'], color=TWIN, label='normal twin')
ax[1].plot(m(df.t), df['position/h-sl-ft'], color=FLIGHT, label='this flight')
ax[1].set(xlabel='time [min]', ylabel='altitude [ft]', title='Altitude departs, recovers after release')
ax[1].legend(fontsize=9)
finish(fig, 'elevstuck_C101.png')

# 4. intermittent thrust loss: engine split
cols = [f'propulsion/engine[{e}]/n1' for e in range(4)]
meta, df, tw = load('C105', cols)
a = meta['anomaly']
e = a['engine'] - 1
fig, ax = plt.subplots(figsize=(10, 3.2))
shade(ax, a)
for k in range(4):
    if k != e:
        ax.plot(m(df.t), df[cols[k]], color=OTHER, lw=0.9, alpha=0.8,
                label='other engines' if k == (0 if e else 1) else None)
ax.plot(m(tw.t), tw[cols[e]], color=TWIN, lw=1.1, label=f'engine {e + 1}, normal twin')
ax.plot(m(df.t), df[cols[e]], color=FLIGHT, label=f'engine {e + 1}, this flight')
ax.set(xlabel='time [min]', ylabel='N1 [%]',
       title=f"Intermittent {a['magnitude'] * 100:.0f} % thrust loss on engine {e + 1}: "
             'each burst splits the engines')
ax.set_xlim((a['onset_s'] - 240) / 60, df.t.iloc[-1] / 60)
ax.set_ylim(68, 96)
ax.legend(fontsize=9, ncol=3, loc='lower left')
finish(fig, 'thrustloss_C105.png')

# 5. sampling: envelopes, magnitude, duration
fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
t = np.linspace(0, 900, 2000)
t0, t1 = 150, 750
step = ((t >= t0) & (t < t1)).astype(float)
ramp = np.where((t >= t0) & (t < t1), (t - t0) / (t1 - t0), 0)
bursts = [(150, 190), (260, 310), (400, 430), (520, 575), (690, 750)]
inter = np.zeros_like(t)
for x, y in bursts:
    inter[(t >= x) & (t < y)] = 1
ax[0].plot(t, step + 2.4, color=FLIGHT, label='step')
ax[0].plot(t, ramp + 1.2, color=TRUTH, label='ramp')
ax[0].plot(t, inter, color=OTHER, label='intermittent')
ax[0].set(yticks=[], xlabel='time [s]', title='Envelope g(t)')
ax[0].legend(fontsize=8.5, loc='center right')
s = np.random.default_rng(1).uniform(size=20000)
lo_, hi_ = 50, 400
ax[1].hist(lo_ * (hi_ / lo_) ** s, bins=np.geomspace(lo_, hi_, 25), color=FLIGHT, alpha=0.8)
ax[1].set_xscale('log')
ax[1].set(xlabel='ALT bias magnitude [ft], log axis', yticks=[], title='Magnitude: log-uniform')
ax[1].set_xticks([50, 100, 200, 400], ['50', '100', '200', '400'])
ax[1].minorticks_off()
d = np.exp(np.random.default_rng(2).normal(np.log(300), 0.6, 20000))
ax[2].hist(np.clip(d, 60, 900), bins=60, color=TRUTH, alpha=0.8)
ax[2].axvline(300, color=INK, lw=1, ls=':')
ax[2].set(xlabel='duration [s]', yticks=[], title='Duration: log-normal, median 300 s')
finish(fig, 'sampling.png')

# 6. normal variability: four normal flights
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
names = {'N100': 'ordinary', 'N101': 'moderate turbulence', 'N102': 'strong wind', 'N103': 'heavy'}
shades = ['#2a78d6', '#6250d6', '#1baf7a', '#eb6834']
for (fid, nm), c in zip(names.items(), shades):
    _, d_, _ = load(fid, ['position/h-sl-ft', 'velocities/vc-kts'])
    ax[0].plot(m(d_.t), d_['position/h-sl-ft'], color=c, lw=1.1, label=f'{fid} {nm}')
    ax[1].plot(m(d_.t), d_['velocities/vc-kts'], color=c, lw=1.0)
ax[0].set(xlabel='time [min]', ylabel='altitude [ft]', title='Every normal flight is different')
ax[0].legend(fontsize=8, loc='lower right')
ax[1].set(xlabel='time [min]', ylabel='CAS [kt]', title='Speed, turbulence and step climbs vary')
finish(fig, 'normals.png')

# 7. effect counts per demo flight
man = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
chans = {c['name']: c for c in json.loads((DATA / 'channels.json').read_text(encoding='utf-8'))}
rows = []
for f in (man if isinstance(man, list) else man['flights']):
    if not f.get('anomaly'):
        continue
    me = json.loads((DATA / 'meta' / f"{f['id']}.json").read_text(encoding='utf-8'))
    sc = [v['score'] for k, v in me['effects'].items() if not chans.get(k, {}).get('context')
          and k not in me['injection_channels']]
    a = me['anomaly']
    tgt = a['variable'] or f"engine {a['engine']}"
    rows.append((f"{f['id']} {a['type'].replace('_', ' ')} · {tgt}",
                 sum(s >= 3 for s in sc), sum(1 <= s < 3 for s in sc), a['group']))
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(8.5, 5.2))
y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], color=FLIGHT, label='strong (score ≥ 3)')
ax.barh(y, [r[2] for r in rows], left=[r[1] for r in rows], color=FLIGHT, alpha=0.35,
        label='moderate (1 ≤ score < 3)')
ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
for lab, r in zip(ax.get_yticklabels(), rows):
    lab.set_color(TRUTH if r[3] == 'control' else INK)
ax.set(xlabel='channels departing from the normal twin')
ax.grid(axis='y', visible=False)
ax.legend(loc='lower right', fontsize=9)
finish(fig, 'effects.png')
