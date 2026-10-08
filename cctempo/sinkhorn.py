"""Debiased Sinkhorn divergence between weighted point clouds.

Uses GeomLoss when it is installed (fast, what the original CC-Tempo/PRESCIENT used) and
falls back to a pure-PyTorch log-domain implementation with epsilon-scaling otherwise.
Cost is |x - y|^2 / 2, matching GeomLoss with p=2.
"""
import math

import torch

try:  # pragma: no cover - optional dependency
    from geomloss import SamplesLoss as _SamplesLoss
except Exception:  # pragma: no cover
    _SamplesLoss = None


def _sqdist(x, y):
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    return (x2 + y2.T - 2.0 * x @ y.T).clamp_min(0.0)


def _eps_schedule(x, y, blur, scaling):
    with torch.no_grad():
        mins = torch.minimum(x.min(0).values, y.min(0).values)
        maxs = torch.maximum(x.max(0).values, y.max(0).values)
        diameter = float((maxs - mins).norm().clamp_min(blur))
    p = 2
    eps = [diameter ** p]
    lo, hi, step = p * math.log(blur), p * math.log(diameter), p * math.log(scaling)
    e = hi
    while e > lo:
        eps.append(math.exp(e))
        e += step
    eps.append(blur ** p)
    return eps


def _softmin(eps, C, h):
    # -eps * log sum_j exp(h_j - C_ij / eps)
    return -eps * torch.logsumexp(h[None, :] - C / eps, dim=1)


def _ot_eps(a, x, b, y, blur, scaling, n_last=1):
    """Entropic OT value with the gradient obtained by a final (differentiable) extrapolation step."""
    loga, logb = a.log(), b.log()
    eps_list = _eps_schedule(x, y, blur, scaling)
    with torch.no_grad():
        Cxy = _sqdist(x, y) / 2.0
        Cyx = Cxy.T
        f = _softmin(eps_list[0], Cxy, logb)
        g = _softmin(eps_list[0], Cyx, loga)
        for eps in eps_list:
            ft = _softmin(eps, Cxy, logb + g / eps)
            gt = _softmin(eps, Cyx, loga + f / eps)
            f, g = 0.5 * (f + ft), 0.5 * (g + gt)
    eps = eps_list[-1]
    Cxy = _sqdist(x, y) / 2.0
    f_new = _softmin(eps, Cxy, logb + g.detach() / eps)
    g_new = _softmin(eps, Cxy.T, loga + f.detach() / eps)
    return (a * f_new).sum() + (b * g_new).sum()


def _ot_sym(a, x, blur, scaling):
    loga = a.log()
    eps_list = _eps_schedule(x, x, blur, scaling)
    with torch.no_grad():
        C = _sqdist(x, x) / 2.0
        f = _softmin(eps_list[0], C, loga)
        for eps in eps_list:
            f = 0.5 * (f + _softmin(eps, C, loga + f / eps))
    eps = eps_list[-1]
    C = _sqdist(x, x) / 2.0
    f_new = _softmin(eps, C, loga + f.detach() / eps)
    return 2.0 * (a * f_new).sum()


def sinkhorn_divergence(a, x, b, y, blur=0.1, scaling=0.7, backend="auto"):
    """S_eps(a.x, b.y) = OT(a,b) - OT(a,a)/2 - OT(b,b)/2 with cost |x-y|^2/2.

    a, b: 1-D nonnegative weights summing to 1 (None -> uniform).
    """
    if a is None:
        a = torch.full((x.shape[0],), 1.0 / x.shape[0], device=x.device, dtype=x.dtype)
    if b is None:
        b = torch.full((y.shape[0],), 1.0 / y.shape[0], device=y.device, dtype=y.dtype)
    a = a / a.sum()
    b = b / b.sum()
    if backend in ("auto", "geomloss") and _SamplesLoss is not None:
        loss = _SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling, debias=True)
        return loss(a, x, b, y)
    if backend == "geomloss":
        raise ImportError("geomloss is not installed")
    return _ot_eps(a, x, b, y, blur, scaling) - 0.5 * _ot_sym(a, x, blur, scaling) - 0.5 * _ot_sym(b, y, blur, scaling)
