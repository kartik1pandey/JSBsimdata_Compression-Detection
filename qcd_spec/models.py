"""Models of the task-aware-compression spec (task_aware_compression.pdf, slides 8-9, 16).

Encoders f (causal, x_{1:t} -> z_t in R^k, LayerNorm on the output):
    'tf'  causal transformer, attention restricted to the last M steps
    'gru' GRU
    'tcn' dilated causal convolutions (receptive field 128)
Decoder g (non-causal, never in the detection path): symmetric conv stack z -> x_hat.
AR likelihood models q(z_{t+1} | z_{t-m+1:t}) with a diagonal Gaussian head (mu, sigma per dimension),
sigma = sigma_min + softplus(.), sigma_min = 1e-2 x the latent std (scale floor, slide 16):
    'tf'  causal transformer, context m = 128 (banded attention mask)   <- the spec's choice
    'gru' GRU (whole history; no finite context)
    'tcn' dilated causal convolutions, receptive field exactly 128 = m
    'lin' linear AR(p = 4) with a learned per-dimension sigma (classical VAR baseline)
mu = z_t + correction (persistence + residual), so every family starts from a sensible forecast.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

M_CTX = 128


def band_mask(T, m, device):
    """Boolean attention mask (True = attend): position t sees t-m+1 .. t."""
    i = torch.arange(T, device=device)
    d = i[:, None] - i[None, :]
    return (d >= 0) & (d < m)


class PosEnc(nn.Module):
    def __init__(self, d, T=4096):
        super().__init__()
        pe = torch.zeros(T, d)
        pos = torch.arange(T).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, h):
        return h + self.pe[:h.shape[1]][None]


class CausalTF(nn.Module):
    def __init__(self, din, dout, d=64, layers=2, heads=4, m=M_CTX):
        super().__init__()
        self.m = m
        self.inp, self.pos = nn.Linear(din, d), PosEnc(d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, dout)

    def forward(self, x):
        T = x.shape[1]
        mask = ~band_mask(T, self.m, x.device)              # True = NOT allowed (PyTorch convention)
        return self.out(self.tf(self.pos(self.inp(x)), mask=mask))


class CausalGRU(nn.Module):
    def __init__(self, din, dout, d=64):
        super().__init__()
        self.inp, self.gru, self.out = nn.Linear(din, d), nn.GRU(d, d, batch_first=True), nn.Linear(d, dout)

    def forward(self, x):
        return self.out(self.gru(F.gelu(self.inp(x)))[0])


class CausalTCN(nn.Module):
    """kernel 2, dilations 1..64 -> receptive field 128."""
    def __init__(self, din, dout, d=64, dil=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = nn.Conv1d(din, d, 1)
        self.convs = nn.ModuleList(nn.Conv1d(d, d, 2, dilation=q) for q in dil)
        self.dil = dil
        self.out = nn.Conv1d(d, dout, 1)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for c, q in zip(self.convs, self.dil):
            h = h + F.gelu(c(F.pad(h, (q, 0))))
        return self.out(h).transpose(1, 2)


def backbone(kind, din, dout, d=64):
    return {'tf': lambda: CausalTF(din, dout, d), 'gru': lambda: CausalGRU(din, dout, d),
            'tcn': lambda: CausalTCN(din, dout, d)}[kind]()


class Encoder(nn.Module):
    """out = 'ln': linear projection followed by LayerNorm(k) (default, every run before 2026-11-17);
    out = 'linear': plain linear output layer (supervisor's suggestion for the k = 100..200 runs)."""
    def __init__(self, kind, D, k, d=64, out='ln'):
        super().__init__()
        self.net = backbone(kind, D, k, d)
        self.ln = nn.LayerNorm(k) if out == 'ln' else nn.Identity()

    def forward(self, x):
        return self.ln(self.net(x))


class Decoder(nn.Module):
    """Non-causal: symmetric convolutions over the whole latent trajectory."""
    def __init__(self, D, k, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(k, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2),
                                 nn.GELU(), nn.Conv1d(d, D, 1))

    def forward(self, z):
        return self.net(z.transpose(1, 2)).transpose(1, 2)


class DecoderMLP(nn.Module):
    """Per-step decoder: x_t from z_t alone (no temporal context)."""
    def __init__(self, D, k, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, D))

    def forward(self, z):
        return self.net(z)


class DecoderBiGRU(nn.Module):
    """Non-causal recurrent decoder: bidirectional GRU over the whole latent trajectory."""
    def __init__(self, D, k, d=64):
        super().__init__()
        self.rnn = nn.GRU(k, d, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * d, D)

    def forward(self, z):
        return self.out(self.rnn(z)[0])


def make_decoder(kind, D, k):
    """kind = 'conv' (default, used by every run before 2026-11-13 afternoon), 'mlp' or 'bigru'."""
    return {'conv': lambda: Decoder(D, k), 'mlp': lambda: DecoderMLP(D, k),
            'bigru': lambda: DecoderBiGRU(D, k)}[kind]()


class ARGauss(nn.Module):
    """q(z_{t+1} | z_{<=t}) as a diagonal Gaussian. forward(z) returns log q of z_{t+1} for t = 0..T-2,
    summed over dimensions: (B, T-1). sig_min (k,) is set by the trainer (detached)."""
    def __init__(self, kind, k, d=64, p=4):
        super().__init__()
        self.kind, self.k, self.p = kind, k, p
        if kind == 'lin':
            self.A = nn.Linear(k * p, k)
            nn.init.zeros_(self.A.weight)
            nn.init.zeros_(self.A.bias)
            self.log_s = nn.Parameter(torch.zeros(k))
        else:
            self.net = backbone(kind, k, 2 * k, d)
        self.register_buffer('sig_min', torch.full((k,), 1e-2))
        self.floor_frac = 0.0

    def params(self, z):
        """mu, sigma predicting z[:, 1:] from contexts ending at z[:, :-1]."""
        ctx = z[:, :-1]
        if self.kind == 'lin':
            lags = [F.pad(ctx, (0, 0, j, 0))[:, :ctx.shape[1]] for j in range(self.p)]   # z_t, z_{t-1}, ...
            mu = ctx + self.A(torch.cat(lags, -1))
            sig = self.sig_min + F.softplus(self.log_s + 0.5413)[None, None].expand_as(mu)
        else:
            o = self.net(ctx)
            mu = ctx + o[..., :self.k]
            sig = self.sig_min + F.softplus(o[..., self.k:] + 0.5413)      # softplus(0.5413) = 1
        return mu, sig

    def forward(self, z):
        mu, sig = self.params(z)
        y = z[:, 1:]
        with torch.no_grad():
            self.floor_frac = float((sig < 1.05 * self.sig_min).float().mean())
        lp = -0.5 * ((y - mu) / sig) ** 2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)
        return lp.sum(-1)


class ARMoG(nn.Module):
    """q(z_{t+1} | z_{<=t}) as a MIXTURE OF MULTIVARIATE GAUSSIANS (supervisor's spec, 2026-11-13).

        q(y) = sum_m pi_m N(y; mu_m, Sigma_m),   Sigma_m = W_m W_m^T + diag(d_m)
    pi, mu, W, d are all outputs of the causal backbone at time t (transformer by default, same banded context
    m = 128). Each component is a full multivariate Gaussian with correlated dimensions; the covariance is
    parameterised as low-rank (rank r) plus diagonal, because a dense k x k Cholesky factor per step and per
    component (k(k+1)/2 = 20,100 numbers at k = 200) does not fit. mu_m = z_t + delta_m (persistence +
    residual), sqrt(d_m) = sigma_min + softplus(.) with the same detached scale floor as ARGauss.
    forward(z) returns log q of z_{t+1} for t = 0..T-2 (summed over dimensions): (B, T-1)."""
    def __init__(self, kind, k, d=64, n_mix=4, rank=8):
        super().__init__()
        self.kind, self.k, self.M, self.r = kind, k, n_mix, rank
        self.net = backbone(kind, k, n_mix * (1 + 2 * k + k * rank), d)
        self.register_buffer('sig_min', torch.full((k,), 1e-2))
        self.floor_frac = 0.0
        self.last_pi = None

    def dist(self, z, f64=False):
        ctx = z[:, :-1]
        B, T, k = ctx.shape
        M, r = self.M, self.r
        o = self.net(ctx)
        logit = o[..., :M]
        rest = o[..., M:].reshape(B, T, M, 2 * k + k * r)
        mu = ctx[:, :, None, :] + rest[..., :k]
        sd = self.sig_min + F.softplus(rest[..., k:2 * k] + 0.5413)
        W = rest[..., 2 * k:].reshape(B, T, M, k, r) * (0.5 / math.sqrt(r))
        with torch.no_grad():
            self.floor_frac = float((sd < 1.05 * self.sig_min).float().mean())
            self.last_pi = torch.softmax(logit, -1).mean((0, 1)).detach().cpu()
        if f64:
            mu, W, sd, logit = mu.double(), W.double(), sd.double(), logit.double()
        comp = torch.distributions.LowRankMultivariateNormal(mu, W, sd ** 2, validate_args=False)
        return torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logit), comp,
                                                     validate_args=False)

    n_f64 = 0   # how many batches needed the float64 fallback (logged; same model, only more precision)

    def forward(self, z):
        # The low-rank Gaussian factorises its k x r capacitance matrix I + W^T D^-1 W with a Cholesky. It is
        # positive-definite mathematically, but in float32 it can fail when the code scale grows and sd is near
        # its floor. Then the same distribution is rebuilt in float64 (identical model, more precision).
        try:
            return self.dist(z).log_prob(z[:, 1:])
        except torch.linalg.LinAlgError:
            ARMoG.n_f64 += 1
            return self.dist(z, f64=True).log_prob(z[:, 1:].double()).float()


def make_q(head, kind, k, d=64, n_mix=4, rank=8):
    """head = 'gauss' (diagonal Gaussian, default) or 'mog' (mixture of multivariate Gaussians)."""
    if head == 'mog':
        return ARMoG(kind, k, d, n_mix, rank)
    return ARGauss(kind, k, d)
