"""Supervisor's spec (2026-11-13): transformer everywhere, q0/q1 with a MIXTURE OF MULTIVARIATE GAUSSIANS head
(4 components, covariance = rank-8 factor + diagonal), compressor with a plain LINEAR output layer, our own code
sizes k (user's choice, 2026-11-17), every loss logged on train and validation each epoch.
Output: results/mog/, checkpoints ~/qcd/ckpt_mog. Same line format as server_jobs.py.

  M1  k = 8, lam 0 / 0.1 / 0.25, seeds 0-2 + raw-data reference (q0/q1 MoG on all 297 channels)
  M2  ablation k = 8, lam 0 / 0.25: MoG head with the old LayerNorm encoder output (separates head vs output layer)
  M3  code size k in {2, 4, 16, 32, 64, 128}, lam 0 / 0.25, seed 0 first, then seeds 1, 2
"""
import os
from pathlib import Path

H = Path(os.environ.get('QCD_HOME', Path.home() / 'qcd'))
C = H / 'code_sup'
E = '--e0 10 --e1 10 --e2 15 --e3 10'
MOG = '--head mog --mix 4 --rank 8'
jobs = []


def job(log, args):
    jobs.append(f'{C / "results/mog"} {H / "ckpt_mog"} {log} spec_run.py {args}')


def cae(k, lam, s, out='linear'):
    job(f'mog_k{k}_lam{lam}_s{s}_{out}', f'--pipeline cae --enc tf --q tf --k {k} --lam {lam} --seed {s} {E} {MOG} '
                                         f'--enc-out {out}')


for s in (0, 1, 2):
    for lam in (0, 0.1, 0.25):
        cae(8, lam, s)
    job(f'mog_raw_s{s}', f'--pipeline raw --q tf --seed {s} --e1 30 {MOG}')
for s in (0, 1, 2):
    for lam in (0, 0.25):
        cae(8, lam, s, out='ln')
for s in (0, 1, 2):
    for k in (2, 4, 16, 32, 64, 128):
        for lam in (0, 0.25):
            cae(k, lam, s)
print('\n'.join(jobs))
