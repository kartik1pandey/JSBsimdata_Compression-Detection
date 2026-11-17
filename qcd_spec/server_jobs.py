"""Job list for the IISc server lane (IISc GPU server): fills every gap left by the laptop runs.

    python server_jobs.py > jobs.txt ;  xargs -P 6 -L 1 -a jobs.txt bash -c   (see run_server.sh)

Each line: <QCD_RES> <QCD_CKPT> <log name> <script> <args...>. Every job skips itself if its result json exists.
Groups (ordered so the most informative runs finish first):
  R  repro     laptop vs server: re-train two laptop runs (results/repro) to check the machines agree
  L  lam 0.1   all three encoder families x 3 seeds (results/srv)          - the gap between lam 0 and 0.25
  Q  bits      quantise every main-grid k = 8 checkpoint (tf, gru, tcn x lam 0, 0.25 x 3 seeds)
  K  k sweep   k in {2,4,16,32,64,128} x tf, gru, tcn x lam 0, 0.25 x 3 seeds   (tf/gru seed 0 already on laptop)
  P  pairing   encoder != density family, encoder tf, gru, tcn x q in {tf, gru, tcn, lin} x lam x 3 seeds
  D  decoder   mlp, bigru x tf, gru, tcn x lam x 3 seeds
  G  long      2x schedule (e0 20 e1 20 e2 30 e3 20; raw e1 60) for the main grid + lam 0.1 -> convergence check
"""
import os
from pathlib import Path

H = Path(os.environ.get('QCD_HOME', Path.home() / 'qcd'))
C = H / 'code'
E = '--e0 10 --e1 10 --e2 15 --e3 10'
E2 = '--e0 20 --e1 20 --e2 30 --e3 20'
FAM = ('tf', 'gru', 'tcn')
SEEDS = (0, 1, 2)
LAPTOP_EXT = {(f, s) for f in ('tf', 'gru') for s in (0,)}     # (encoder family, seed) already done in results/ext
jobs = []


def job(res, ckpt, log, script, args):
    jobs.append(f'{C / res} {H / ckpt} {log} {script} {args}')


def cae(res, ckpt, log, enc, q, k, lam, s, ex=E, dec='conv'):
    d = '' if dec == 'conv' else f' --dec {dec}'
    job(res, ckpt, log, 'spec_run.py', f'--pipeline cae --enc {enc} --q {q} --k {k} --lam {lam} --seed {s} {ex}{d}')


# R
cae('results/repro', 'ckpt_repro', 'repro_tcn_lam0.25', 'tcn', 'tcn', 8, 0.25, 0)
job('results/repro', 'ckpt_repro', 'repro_raw_tcn', 'spec_run.py', '--pipeline raw --q tcn --seed 0 --e1 30')
# L
for s in SEEDS:
    for f in FAM:
        cae('results/srv', 'ckpt_srv', f'lam0.1_{f}_s{s}', f, f, 8, 0.1, s)
# Q (laptop checkpoints of the main grid)
for s in SEEDS:
    for f in FAM:
        for lam in (0, 0.25):
            t = f'cae_e{f}_q{f}_k8_lam{lam:g}_s{s}'
            if (f, s) in LAPTOP_EXT:
                continue
            job('results/srv', 'ckpt', f'quant_{t}', 'quant_eval.py', f'--tag {t} --bits 1 2 3 4 6 8 --res-main {C / "results"}')
# K, then P, then D: seed 0 first (fills the TCN gaps), then seeds 1, 2
for s in SEEDS:
    for f in FAM:
        if (f, s) in LAPTOP_EXT:
            continue
        for k in (2, 4, 16, 32, 64, 128):
            for lam in (0, 0.25):
                cae('results/srv', 'ckpt_srv', f'k{k}_{f}_lam{lam}_s{s}', f, f, k, lam, s)
for s in SEEDS:
    for f in FAM:
        if (f, s) in LAPTOP_EXT:
            continue
        for q in ('tf', 'gru', 'tcn', 'lin'):
            if q == f:
                continue
            for lam in (0, 0.25):
                cae('results/srv', 'ckpt_srv', f'pair_e{f}_q{q}_lam{lam}_s{s}', f, q, 8, lam, s)
        for dec in ('mlp', 'bigru'):
            for lam in (0, 0.25):
                cae('results/srv', 'ckpt_srv', f'dec_{dec}_{f}_lam{lam}_s{s}', f, f, 8, lam, s, dec=dec)
# S  extra seeds 3..7 for the key k = 8 settings: the repro check showed the sign check can flip between two
#    trainings of the same seed (GPU non-determinism), so validity needs more than 3 seeds to be estimated
for s in range(3, 8):
    for q in ('tf', 'gru', 'tcn', 'lin'):
        job('results/srv', 'ckpt_srv', f'raw_q{q}_s{s}', 'spec_run.py', f'--pipeline raw --q {q} --seed {s} --e1 30')
    for f in FAM:
        for lam in (0, 0.1, 0.25):
            cae('results/srv', 'ckpt_srv', f'seed_{f}_lam{lam}_s{s}', f, f, 8, lam, s)
# G
for s in SEEDS:
    for q in ('tf', 'gru', 'tcn', 'lin'):
        job('results/long', 'ckpt_long', f'raw_q{q}_s{s}', 'spec_run.py', f'--pipeline raw --q {q} --seed {s} --e1 60')
    for f in FAM:
        for lam in (0, 0.1, 0.25, 0.5):
            cae('results/long', 'ckpt_long', f'cae_{f}_lam{lam}_s{s}', f, f, 8, lam, s, ex=E2)

print('\n'.join(jobs))
