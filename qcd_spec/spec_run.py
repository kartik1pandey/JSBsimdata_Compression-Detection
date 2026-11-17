"""One experiment of the task-aware-compression spec (task_aware_compression.pdf), end to end:
train -> calibrate h on val_normal -> verify FA on test_normal -> detect on test_change -> metrics.

    python spec_run.py --pipeline raw --q tf --seed 0
    python spec_run.py --pipeline cae --enc tf --q tf --k 8 --lam 0.5 --seed 0

Pipelines (slide 2): raw (q0, q1 on x), cae with lambda = 0 (unaware) or lambda > 0 (task-aware).
Objective (slide 15), encoder f only:  L = (1 - lam) D + lam T + eta H,  eta = 0.1
    D = L_rec / sigma_X^2,  L_rec = 1/2 sum_c (1/n_c) sum_i ||X - g(f(X))||^2 / (D T)   (both classes)
    T = -log max(J_ema, eps) with the gradient through J_t only:  dT = -dJ_t / J_ema   (slide 19)
    J = 1/(n1 (T - t0)) sum_{j, t > t0} [log q1(z_t; sg th1) - log q0(z_t; sg th0)]     on D1 only
    H = (L0 + L1) / (k T),  L0, L1 = per-trajectory NLL totals of q0 on D0, q1 on D1
  decoder g: min D;  q0: min L0 on D0 only;  q1: min L1 on D1 only (z detached: pure MLE).
Schedule (slide 19): stage 0 lam = eta = 0 (min D) | stage 1 f frozen, fit q0, q1, check J > 0 |
  stage 2 lam ramps 0 -> target over N_WARM epochs; per iteration N_INNER steps on (q0, q1), 1 step on (f, g) |
  stage 3 (added here, same for every lambda): f frozen, refit q0, q1 so the densities match the final code.
Calibration (slides 20-21): M_i = max_t W_t on val_normal, h = M_(r), r = ceil((N + 1)(1 - alpha)).
Scoring (slide 24): T < nu false alarm, nu <= T detection (delay T - nu), no alarm = miss; mean and median.
LLR index offset: s[j] scores the sample j + 1, so the stopping time is j + 1.
"""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from models import Encoder, make_decoder, make_q

DATA = Path(os.environ.get('QCD_DATA', r'C:\Users\karti\IISc_data\qcd_pure'))
CKPT = Path(os.environ.get('QCD_CKPT', r'C:\Users\karti\IISc_data\qcd_spec_ckpt'))
RES = Path(os.environ.get('QCD_RES', str(Path(__file__).resolve().parent / 'results')))
DEV = 'cuda'
T0, ALPHA, ETA, EPS, HZ = 40, 0.05, 0.1, 1e-3, 4
B = 8                                   # trajectories per class per minibatch (both classes every batch)
N_INNER, N_WARM, EMA_BETA = 3, 5, 0.1
TRACE = {}                              # per-iteration loss traces, saved to <RES>/live/<tag>_trace.npz


def mm(role):
    return np.load(DATA / f'{role}_x.npy', mmap_mode='r')


def meta(role):
    return dict(np.load(DATA / f'{role}_meta.npz'))


def batch(x, rng, n=B):
    idx = np.sort(rng.choice(len(x), n, replace=False))
    return torch.from_numpy(np.ascontiguousarray(x[idx])).to(DEV)


def sigma2_x():
    p = DATA / 'sigma2_x.json'
    if p.exists():
        return json.loads(p.read_text())['sigma2_x']
    s1 = s2 = 0.
    n = 0
    for r in ('train_normal', 'train_abnormal'):
        x = mm(r)
        for i in range(0, len(x), 40):
            v = np.asarray(x[i:i + 40], dtype=np.float64).reshape(-1, x.shape[-1])
            s1, s2, n = s1 + v.sum(0), s2 + (v ** 2).sum(0), n + len(v)
    var = float((s2 / n - (s1 / n) ** 2).mean())
    p.write_text(json.dumps(dict(sigma2_x=var, rule='mean over channels of the variance of X about its mean, '
                                                 'train_normal + train_abnormal')))
    return var


class Pipe:
    def __init__(self, a, D):
        self.a, self.raw = a, a.pipeline == 'raw'
        self.k = D if self.raw else a.k
        g = lambda n, v: getattr(a, n, v)                     # older result files lack the newer options
        if not self.raw:
            self.f = Encoder(a.enc, D, a.k, g('d', 64), g('enc_out', 'ln')).to(DEV)
            self.g = make_decoder(g('dec', 'conv'), D, a.k).to(DEV)
            self.opt_fg = torch.optim.Adam(list(self.f.parameters()) + list(self.g.parameters()), a.lr)
        self.q0, self.q1 = (make_q(g('head', 'gauss'), a.q, self.k, g('d', 64), g('mix', 4), g('rank', 8)).to(DEV)
                            for _ in range(2))
        self.opt_q = torch.optim.Adam(list(self.q0.parameters()) + list(self.q1.parameters()), a.lr)

    def enc(self, x):
        return x if self.raw else self.f(x)

    @torch.no_grad()
    def set_floor(self, x0, x1):
        """sigma_min = 1e-2 x per-dimension latent std (raw: std of x), on a fresh two-class batch."""
        z = torch.cat([self.enc(x0), self.enc(x1)]).reshape(-1, self.k)
        s = 1e-2 * z.std(0).clamp_min(1e-3)
        self.q0.sig_min.copy_(s)
        self.q1.sig_min.copy_(s)

    def q_step(self, x0, x1):
        with torch.no_grad():
            z0, z1 = self.enc(x0), self.enc(x1)
        n = self.k * (z0.shape[1] - 1)
        l0 = -self.q0(z0).sum(1).mean() / n                  # = L0 / (k T): same minimiser, O(1) scale
        l1 = -self.q1(z1).sum(1).mean() / n
        self.opt_q.zero_grad()
        (l0 + l1).backward()
        gn = torch.nn.utils.clip_grad_norm_(list(self.q0.parameters()) + list(self.q1.parameters()), 1.0)
        self.opt_q.step()
        return float(l0), float(l1), float(gn)


def dbar(xh, x, s2):
    return ((xh - x) ** 2).mean() / s2


def train(a, P, s2, log):
    rng = np.random.default_rng(a.seed)
    x0, x1 = mm('train_normal'), mm('train_abnormal')
    it = len(x0) // B
    t_start = time.time()

    live = RES / 'live' / f'{a.tag}.jsonl'                   # one JSON line per epoch, readable while training
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_text(json.dumps(dict(stage='args', **vars(a))) + '\n')

    def ep_log(stage, ep, **kw):
        kw.update(val_losses(P, s2, kw.get('lam', a.lam if stage == 's3' else 0.0)))
        row = dict(stage=stage, epoch=ep, t=round(time.time() - t_start, 1), **kw)
        log.append(row)
        with open(live, 'a') as fh:
            fh.write(json.dumps(row, default=float) + '\n')
        print(stage, ep, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()}, flush=True)

    if not P.raw:                                             # stage 0: warm-start the code, min D
        for ep in range(a.e0):
            ds, gs = [], []
            for _ in range(it):
                b0, b1 = batch(x0, rng), batch(x1, rng)
                d = 0.5 * (dbar(P.g(P.f(b0)), b0, s2) + dbar(P.g(P.f(b1)), b1, s2))
                P.opt_fg.zero_grad()
                d.backward()
                gn = torch.nn.utils.clip_grad_norm_(list(P.f.parameters()) + list(P.g.parameters()), 1.0)
                P.opt_fg.step()
                ds.append(float(d))
                gs.append(float(gn))
                TRACE.setdefault('s0_D', []).append(float(d))
            ep_log('s0', ep, D=float(np.mean(ds)), loss=float(np.mean(ds)), gn_fg=float(np.mean(gs)), lr_fg=a.lr)

    def fit_q(stage, epochs, decay=False):
        """decay: cosine learning-rate decay to 0 over the stage (the FINAL density fit of every pipeline),
        so q0 and q1 settle instead of bouncing at a constant step size (the LLR is their difference)."""
        n_tot, k_it = epochs * it, 0
        for ep in range(epochs):
            P.set_floor(batch(x0, rng), batch(x1, rng))
            ls = []
            for _ in range(it):
                if decay:
                    for gr in P.opt_q.param_groups:
                        gr['lr'] = a.lr * 0.5 * (1 + math.cos(math.pi * k_it / n_tot))
                    k_it += 1
                ls.append(P.q_step(batch(x0, rng), batch(x1, rng)))
                TRACE.setdefault(f'{stage}_L0', []).append(ls[-1][0])
                TRACE.setdefault(f'{stage}_L1', []).append(ls[-1][1])
            l0m, l1m = float(np.mean([v[0] for v in ls])), float(np.mean([v[1] for v in ls]))
            ep_log(stage, ep, L0=l0m, L1=l1m, Hq=l0m + l1m,
                   gn_q=float(np.mean([v[2] for v in ls])), lr_q=P.opt_q.param_groups[0]['lr'], **monitor(P))

    fit_q('s1', a.e1, decay=P.raw)                            # stage 1 (raw: the whole training)
    if P.raw:
        return
    j_ema = None
    for ep in range(a.e2):                                    # stage 2: alternate
        lam = a.lam * min(1.0, (ep + 1) / N_WARM)
        P.set_floor(batch(x0, rng), batch(x1, rng))
        rec = dict(D=[], T=[], H=[], J=[], loss=[], L0=[], L1=[], gn_fg=[], gn_q=[])
        for _ in range(it):
            for _ in range(N_INNER):
                l0v, l1v, gq = P.q_step(batch(x0, rng), batch(x1, rng))
                rec['gn_q'].append(gq)
                rec['L0'].append(l0v)
                rec['L1'].append(l1v)
            b0, b1 = batch(x0, rng), batch(x1, rng)
            for q in (P.q0, P.q1):
                q.requires_grad_(False)                       # stop-gradient on theta0, theta1 (slide 14)
            z0, z1 = P.f(b0), P.f(b1)
            d = 0.5 * (dbar(P.g(z0), b0, s2) + dbar(P.g(z1), b1, s2))
            d_dec = 0.5 * (dbar(P.g(z0.detach()), b0, s2) + dbar(P.g(z1.detach()), b1, s2))
            lq0n, lq1a = P.q0(z0), P.q1(z1)
            n = P.k * (z0.shape[1] - 1)
            H = (-lq0n.sum(1).mean() - lq1a.sum(1).mean()) / n
            loss = (1 - lam) * d + ETA * H + lam * d_dec       # decoder always gets the full grad of D
            if a.lam > 0:
                J = (lq1a - P.q0(z1))[:, T0:].mean()           # per-step drift on D1, t > t0
                jv = float(J)
                j_ema = jv if j_ema is None else (1 - EMA_BETA) * j_ema + EMA_BETA * jv
                Tterm = -J / max(j_ema, EPS)                   # grad of -log J with an EMA denominator
                loss = loss + lam * Tterm
                rec['J'].append(jv)
                rec['T'].append(-math.log(max(jv, EPS)))
            # reported combined objective L = (1 - lam) D + lam T + eta H with T = -log max(J, eps) (the value,
            # not the EMA surrogate whose gradient is used above)
            rec['loss'].append((1 - lam) * float(d) + ETA * float(H)
                               + (lam * rec['T'][-1] if a.lam > 0 else 0.0))
            P.opt_fg.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(P.f.parameters()) + list(P.g.parameters()), 1.0)
            P.opt_fg.step()
            for q in (P.q0, P.q1):
                q.requires_grad_(True)
            rec['D'].append(float(d))
            rec['H'].append(float(H))
            rec['gn_fg'].append(float(gn))
            TRACE.setdefault('s2_D', []).append(float(d))
            TRACE.setdefault('s2_H', []).append(float(H))
            TRACE.setdefault('s2_loss', []).append(float(loss))
            if a.lam > 0:
                TRACE.setdefault('s2_J', []).append(jv)
                TRACE.setdefault('s2_J_ema', []).append(j_ema)
        ep_log('s2', ep, lam=lam, j_ema=j_ema, **{k: float(np.mean(v)) if v else None for k, v in rec.items()},
               **monitor(P))
    for gr in P.opt_q.param_groups:
        gr['lr'] = a.lr
    fit_q('s3', a.e3, decay=True)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def scores(P, role, bs=10, want_d=False, s2=1.0):
    """log q0, log q1 of every next sample (N, T-1), and per-trajectory D for the set."""
    x = mm(role)
    l0, l1, dd = [], [], []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(DEV)
        z = P.enc(xb)
        l0.append(P.q0(z).cpu().numpy())
        l1.append(P.q1(z).cpu().numpy())
        if want_d and not P.raw:
            dd.append(((P.g(z) - xb) ** 2).mean((1, 2)).cpu().numpy() / s2)
    return np.concatenate(l0), np.concatenate(l1), (np.concatenate(dd) if dd else None)


@torch.no_grad()
def monitor(P, n=24):
    """Cheap per-epoch diagnostics on fixed validation subsets (slide 25)."""
    out = {}
    for role, key in (('val_normal', 'E0s'), ('val_abnormal', 'E1s')):
        x = mm(role)
        xb = torch.from_numpy(np.ascontiguousarray(x[:n])).to(DEV)
        z = P.enc(xb)
        s = (P.q1(z) - P.q0(z))[:, T0:].cpu().numpy()
        out[key] = float(s.mean())
        if role == 'val_normal':
            out['margin0'] = float(s.mean() / (s.std() + 1e-12))
            out['L0val'] = float(-P.q0(z).mean() / P.k)
    out['floor_frac'] = max(P.q0.floor_frac, P.q1.floor_frac)
    return out


@torch.no_grad()
def val_losses(P, s2, lam, bs=10):
    """Every loss term on the FULL validation sets (val_normal = D0, val_abnormal = D1), with the same definitions
    and normalisations as the training terms, so train and validation curves are directly comparable:
      val_D    reconstruction error D (both classes, MSE / sigma_X^2)            (compressed pipelines only)
      val_L0   NLL of q0 on val_normal per number per step;  val_L1  NLL of q1 on val_abnormal
      val_H    = val_L0 + val_L1 (the modellability term, slide 15)
      val_J    drift on val_abnormal, t > t0;  val_T = -log max(val_J, eps)
      val_loss = (1 - lam) val_D + lam val_T + eta val_H   (the combined objective at the current lam)
      z_std    mean per-dimension std of the code (watch for code-scale collapse with a linear encoder output)
      pi_max   largest average mixture weight of q0 (mixture head only; 1 = a single component is used)"""
    mods = [P.q0, P.q1] + ([] if P.raw else [P.f, P.g])
    for mod in mods:
        mod.eval()
    acc = dict(D=[], L0=[], L1=[], J=[], zs=[])
    for role in ('val_normal', 'val_abnormal'):
        x = mm(role)
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(DEV)
            z = P.enc(xb)
            n = P.k * (z.shape[1] - 1)
            if not P.raw:
                acc['D'].append(float(dbar(P.g(z), xb, s2)))
                acc['zs'].append(float(z.reshape(-1, P.k).std(0).mean()))
            if role == 'val_normal':
                acc['L0'].append(float(-P.q0(z).sum(1).mean() / n))
            else:
                l1 = P.q1(z)
                acc['L1'].append(float(-l1.sum(1).mean() / n))
                acc['J'].append(float((l1 - P.q0(z))[:, T0:].mean()))
    for mod in mods:
        mod.train()
    o = {f'val_{k}': float(np.mean(v)) for k, v in acc.items() if v and k != 'zs'}
    o['val_H'] = o['val_L0'] + o['val_L1']
    o['val_T'] = -math.log(max(o['val_J'], EPS))
    o['val_loss'] = (1 - lam) * o.get('val_D', 0.0) + lam * o['val_T'] + ETA * o['val_H']
    if acc['zs']:
        o['z_std'] = float(np.mean(acc['zs']))
    if getattr(P.q0, 'last_pi', None) is not None:
        o['pi_max'] = float(P.q0.last_pi.max())
    return o


def cusum(s, start=T0 + 1):
    """W_t = max(0, W_{t-1} + s_t) for t >= start (W = 0 before), per trajectory. (N, T-1)."""
    W = np.zeros_like(s)
    w = np.zeros(len(s))
    for t in range(start, s.shape[1]):
        w = np.maximum(0.0, w + s[:, t])
        W[:, t] = w
    return W


def calibrate(s_val):
    M = cusum(s_val).max(1)
    N = len(M)
    r = math.ceil((N + 1) * (1 - ALPHA))
    ok = r <= N                                    # needs N >= 1/alpha - 1 = 19 (slide 21)
    return float(np.sort(M)[min(r, N) - 1]), dict(N=N, r=r, valid=ok,
                                                  plain_quantile=float(np.quantile(M, 1 - ALPHA)))


def detect(s, h, nu):
    W = cusum(s)
    hit = W >= h
    hit[:, :T0 + 1] = False
    first = np.where(hit.any(1), hit.argmax(1), -1)
    stop = np.where(first >= 0, first + 1, -1)                 # s[j] scores sample j + 1
    fa = (stop >= 0) & (stop < nu)
    det = (stop >= 0) & (stop >= nu)
    delay = (stop[det] - nu[det]) / HZ
    return dict(DR=float(det.mean()), FA_pre=float(fa.mean()), miss=float((stop < 0).mean()),
                ADD_mean_s=float(delay.mean()) if det.any() else None,
                ADD_median_s=float(np.median(delay)) if det.any() else None, n=int(len(s))), stop


def evaluate(P, s2):
    S = {r: scores(P, r, want_d=r in ('test_normal', 'test_abnormal'), s2=s2)
         for r in ('val_normal', 'val_abnormal', 'test_normal', 'test_abnormal', 'test_change', 'val_change')}
    llr = {r: v[1] - v[0] for r, v in S.items()}
    m = {}
    # cross-entropies per step (nats/step), t > t0, on pure test data (slide 23)
    for split in ('val', 'test'):
        l0n, l1n, _ = S[f'{split}_normal']
        l0a, l1a, _ = S[f'{split}_abnormal']
        ce = dict(CE00=-l0n[:, T0:].mean(), CE01=-l1n[:, T0:].mean(), CE10=-l0a[:, T0:].mean(),
                  CE11=-l1a[:, T0:].mean())
        ce = {k: float(v) for k, v in ce.items()}
        ce['K'] = ce['CE10'] - ce['CE11']
        ce['signs_ok'] = bool(ce['CE10'] > ce['CE11'] and ce['CE01'] > ce['CE00'])
        sn = llr[f'{split}_normal'][:, T0:]
        ce['E0s'], ce['E1s'] = float(sn.mean()), float(llr[f'{split}_abnormal'][:, T0:].mean())
        ce['margin0'] = float(sn.mean() / sn.std())
        ce['J_eff'] = float(llr[f'{split}_abnormal'][:, T0:].mean() ** 2 / llr[f'{split}_abnormal'][:, T0:].var())
        m[split] = ce
    if not P.raw:
        m['D_test_normal'] = float(S['test_normal'][2].mean())
        m['D_test_abnormal'] = float(S['test_abnormal'][2].mean())
    # calibration on val_normal only, FA check on test_normal, detection on test_change
    h, cal = calibrate(llr['val_normal'])
    m['h'], m['calibration'] = h, cal
    m['FA_test_normal'] = float((cusum(llr['test_normal']).max(1) > h).mean())
    nu = meta('test_change')['change_point'].astype(int)
    m['detect'], stop = detect(llr['test_change'], h, nu)
    types = meta('test_change')['anomaly_type'].astype(str)
    per = {}
    for t in sorted(set(types)):
        w = types == t
        d, _ = detect(llr['test_change'][w], h, nu[w])
        per[t] = d
    m['detect_per_type'] = per
    # variance-normalised diagnostic (slide 22)
    sv = llr['val_normal'][:, T0:]
    mu0, sd0 = float(sv.mean()), float(sv.std())
    hv, _ = calibrate((llr['val_normal'] - mu0) / sd0)
    m['varnorm'] = dict(mu0=mu0, sd0=sd0, h=hv, FA_test_normal=float((cusum((llr['test_normal'] - mu0) / sd0)
                                                                        .max(1) > hv).mean()),
                        detect=detect((llr['test_change'] - mu0) / sd0, hv, nu)[0])
    # post-change drift curve on val_change (measurement only, slide 17)
    nv = meta('val_change')['change_point'].astype(int)
    s = llr['val_change']
    rel = np.arange(s.shape[1])[None, :] + 1 - nv[:, None]
    curve = {}
    for lo in range(-40, 440, 20):
        w = (rel >= lo) & (rel < lo + 20)
        if w.any():
            curve[lo] = float(s[w].mean())
    m['drift_curve_val_change'] = curve
    m['J_pure_val_abnormal'] = m['val']['E1s']
    return m, llr, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', choices=('raw', 'cae'), required=True)
    ap.add_argument('--enc', default='tf')
    ap.add_argument('--q', default='tf')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--e0', type=int, default=10)
    ap.add_argument('--e1', type=int, default=10)
    ap.add_argument('--e2', type=int, default=15)
    ap.add_argument('--e3', type=int, default=5)
    ap.add_argument('--dec', default='conv', choices=('conv', 'mlp', 'bigru'))
    ap.add_argument('--head', default='gauss', choices=('gauss', 'mog'), help='q0/q1 output head')
    ap.add_argument('--mix', type=int, default=4, help='mixture components (head mog)')
    ap.add_argument('--rank', type=int, default=8, help='low-rank covariance factor per component (head mog)')
    ap.add_argument('--enc-out', default='ln', choices=('ln', 'linear'), help='encoder output layer')
    ap.add_argument('--d', type=int, default=64, help='hidden width of encoder and q0/q1 backbones')
    a = ap.parse_args()
    tag = (f'raw_q{a.q}_s{a.seed}' if a.pipeline == 'raw'
           else f'cae_e{a.enc}_q{a.q}_k{a.k}_lam{a.lam:g}_s{a.seed}' + ('' if a.dec == 'conv' else f'_d{a.dec}'))
    tag += ('' if a.head == 'gauss' else f'_hmog{a.mix}r{a.rank}') + ('' if a.enc_out == 'ln' else '_olin') \
        + ('' if a.d == 64 else f'_w{a.d}')
    a.tag = tag
    RES.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    if (RES / f'{tag}.json').exists():
        print('skip', tag)
        return
    lock = RES / 'running' / f'{tag}.lock'                 # two runners may share one job list: claim the tag
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        print('skip (running in another runner)', tag)
        return
    import atexit
    atexit.register(lambda: lock.unlink(missing_ok=True))   # released on normal exit and on errors (e.g. OOM)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    s2 = sigma2_x()
    D = mm('train_normal').shape[-1]
    P = Pipe(a, D)
    log = []
    t0 = time.time()
    train(a, P, s2, log)
    train_s = time.time() - t0
    for mod in ([P.q0, P.q1] + ([] if P.raw else [P.f, P.g])):
        mod.eval()
    m, llr, stop = evaluate(P, s2)
    mods = {n: getattr(P, n) for n in ('f', 'g', 'q0', 'q1') if getattr(P, n, None) is not None}
    config = dict(params={n: int(sum(p.numel() for p in v.parameters())) for n, v in mods.items()},
                  arch={n: str(v) for n, v in mods.items()},
                  hyper=dict(lr=a.lr, optimizer='Adam', batch_per_class=B, inner_q_steps=N_INNER,
                             warm_epochs=N_WARM, eta=ETA, ema_beta=EMA_BETA, t0=T0, alpha=ALPHA, grad_clip=1.0,
                             epochs=dict(s0=a.e0, s1=a.e1, s2=a.e2, s3=a.e3),
                             iters_per_epoch=len(mm('train_normal')) // B,
                             final_q_lr='cosine decay to 0 in the last fit stage',
                             device=torch.cuda.get_device_name(0)),
                  gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20))
    m.update(tag=tag, args=vars(a), train_s=train_s, sigma2_x=s2, D=D, log=log, config=config)
    np.savez_compressed(RES / 'live' / f'{tag}_trace.npz', **{k: np.asarray(v, dtype=np.float32)
                                                             for k, v in TRACE.items()})
    torch.save({n: getattr(P, n).state_dict() for n in ('q0', 'q1', 'f', 'g') if getattr(P, n, None) is not None},
               CKPT / f'{tag}.pt')
    np.savez_compressed(CKPT / f'{tag}_llr.npz', **llr, stop=stop)
    (RES / f'{tag}.json').write_text(json.dumps(m, indent=1, default=float))
    row = dict(tag=tag, pipeline=a.pipeline, enc=a.enc if a.pipeline == 'cae' else '', dec=a.dec, q=a.q,
               k=a.k if a.pipeline == 'cae' else D, lam=a.lam if a.pipeline == 'cae' else '', seed=a.seed,
               D_test_normal=m.get('D_test_normal', 0.0), D_test_abnormal=m.get('D_test_abnormal', 0.0),
               K_val=m['val']['K'], K_test=m['test']['K'], signs_ok=m['test']['signs_ok'],
               E0s=m['test']['E0s'], E1s=m['test']['E1s'], margin0=m['test']['margin0'], h=m['h'],
               FA_test_normal=m['FA_test_normal'], DR=m['detect']['DR'], FA_pre=m['detect']['FA_pre'],
               miss=m['detect']['miss'], ADD_mean_s=m['detect']['ADD_mean_s'],
               ADD_median_s=m['detect']['ADD_median_s'], train_s=round(train_s))
    p = RES / 'runs.csv'
    new = not p.exists()
    cols = list(row) if new else next(csv.reader(open(p)))     # keep an existing file's columns aligned
    with open(p, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)
    print('RESULT', json.dumps(row, default=float), flush=True)


if __name__ == '__main__':
    main()
