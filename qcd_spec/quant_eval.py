"""Bit budget: quantise a trained code z = f(x) to b bits per latent number and re-run the whole evaluation.

    python quant_eval.py --tag cae_etf_qtf_k8_lam0.25_s0 --bits 1 2 3 4 6 8

Rate on the link = k numbers x b bits x 4 Hz  (raw float32: 297 x 32 x 4 = 38,016 bit/s).
Quantiser: uniform, per latent dimension, 2^b levels spanning the 0.1..99.9 percentile range of the codes of
train_normal + train_abnormal (fit on TRAIN only), values outside are clipped to the end levels.
Two receivers per b:
  frozen  q0, q1 from the checkpoint applied to the quantised code (no retraining);
  refit   q0, q1 re-fitted on the quantised code of the TRAIN sets (cosine lr decay, --refit-epochs), because a
          receiver only ever sees quantised values. The encoder and decoder stay frozen. During the refit a
          uniform dither of one quantisation step is added, so the Gaussian learns the bin mass instead of
          collapsing onto the grid points; evaluation uses the exact quantised values (bin width cancels in the LLR).
h is recalibrated on (quantised) val_normal for every b and receiver, exactly as in spec_run.evaluate.
Results: <QCD_RES>/quant/<tag>_b<b>_<receiver>.json (default QCD_RES = results/ext).
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('QCD_RES', str(Path(__file__).resolve().parent / 'results' / 'ext'))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import spec_run as S  # noqa: E402

HZ, D_RAW = 4, 297


def load_pipe(tag, res_main):
    m = json.loads((res_main / f'{tag}.json').read_text())
    a = SimpleNamespace(**{**dict(dec='conv'), **m['args']})
    D = S.mm('train_normal').shape[-1]
    P = S.Pipe(a, D)
    sd = torch.load(S.CKPT / f'{tag}.pt', map_location=S.DEV)
    for n in ('q0', 'q1', 'f', 'g'):
        getattr(P, n).load_state_dict(sd[n])
    for n in ('q0', 'q1', 'f', 'g'):
        getattr(P, n).eval()
    return P, a, m


@torch.no_grad()
def code_range(P, n=80):
    zs = []
    for r in ('train_normal', 'train_abnormal'):
        x = S.mm(r)
        for i in range(0, n, 10):
            zs.append(P.f(torch.from_numpy(np.ascontiguousarray(x[i:i + 10])).to(S.DEV)).reshape(-1, P.k).cpu())
    z = torch.cat(zs)
    return torch.quantile(z, 0.001, dim=0).to(S.DEV), torch.quantile(z, 0.999, dim=0).to(S.DEV)


def make_quant(lo, hi, b):
    L = 2 ** b - 1
    step = (hi - lo).clamp_min(1e-6) / L

    def q(z):
        return lo + torch.round(((z - lo) / step).clamp(0, L)) * step
    return q, step


def refit(P, a, epochs):
    rng = np.random.default_rng(a.seed + 1000)
    x0, x1 = S.mm('train_normal'), S.mm('train_abnormal')
    it = len(x0) // S.B
    n_tot, k_it, log = epochs * it, 0, []
    for mod in (P.q0, P.q1):
        mod.train()
    for ep in range(epochs):
        P.set_floor(S.batch(x0, rng), S.batch(x1, rng))
        ls = []
        for _ in range(it):
            for gr in P.opt_q.param_groups:
                gr['lr'] = a.lr * 0.5 * (1 + math.cos(math.pi * k_it / n_tot))
            k_it += 1
            ls.append(P.q_step(S.batch(x0, rng), S.batch(x1, rng)))
        log.append(dict(stage='refit', epoch=ep, L0=float(np.mean([v[0] for v in ls])),
                        L1=float(np.mean([v[1] for v in ls])), **S.monitor(P)))
        print('refit', ep, {k: round(v, 4) for k, v in log[-1].items() if isinstance(v, float)}, flush=True)
    for mod in (P.q0, P.q1):
        mod.eval()
    return log


def slim(m):
    return {k: m[k] for k in ('val', 'test', 'D_test_normal', 'D_test_abnormal', 'h', 'calibration',
                              'FA_test_normal', 'detect', 'detect_per_type') if k in m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--bits', type=int, nargs='+', default=[1, 2, 3, 4, 6, 8])
    ap.add_argument('--refit-epochs', type=int, default=5)
    ap.add_argument('--res-main', default=str(Path(__file__).resolve().parent / 'results'))
    a_cli = ap.parse_args()
    res_main = Path(a_cli.res_main)
    out = S.RES / 'quant'
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    s2 = S.sigma2_x()
    P, a, m_float = load_pipe(a_cli.tag, res_main)
    lo, hi = code_range(P)
    q_state = {n: copy.deepcopy(getattr(P, n).state_dict()) for n in ('q0', 'q1')}
    for b in a_cli.bits:
        qz, step = make_quant(lo, hi, b)
        enc_eval = (lambda x, qz=qz: qz(P.f(x)))
        enc_fit = (lambda x, qz=qz, st=step: (lambda z: z + (torch.rand_like(z) - 0.5) * st)(qz(P.f(x))))
        for recv in ('frozen', 'refit'):
            p = out / f'{a_cli.tag}_b{b}_{recv}.json'
            if p.exists():
                print('skip', p.name)
                continue
            for n in ('q0', 'q1'):
                getattr(P, n).load_state_dict(q_state[n])
            P.opt_q = torch.optim.Adam(list(P.q0.parameters()) + list(P.q1.parameters()), a.lr)
            t0 = time.time()
            P.enc = enc_fit                      # dithered: the density learns bins, not grid points
            log = refit(P, a, a_cli.refit_epochs) if recv == 'refit' else []
            P.enc = enc_eval                     # evaluation on the exact quantised values
            with torch.no_grad():
                m, _, _ = S.evaluate(P, s2)
            r = dict(tag=a_cli.tag, bits=b, receiver=recv, k=P.k, rate_bit_s=P.k * b * HZ,
                     raw_rate_bit_s=D_RAW * 32 * HZ, args=vars(a), log=log, secs=round(time.time() - t0, 1),
                     **slim(m))
            p.write_text(json.dumps(r, indent=1, default=float))
            d = m['detect']
            print(f'RESULT b={b} {recv}: rate={P.k * b * HZ} bit/s K={m["test"]["K"]:.2f} '
                  f'signs={m["test"]["signs_ok"]} FA={m["FA_test_normal"]:.3f} DR={d["DR"]:.3f} '
                  f'ADDmed={d["ADD_median_s"]}', flush=True)


if __name__ == '__main__':
    main()
