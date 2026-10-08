"""Waddington-OT style baseline (Schiebinger et al. 2019) with POT.

Unbalanced entropic OT couplings between consecutive time points (source mass ~ growth),
fate probabilities by pulling back the final-time cell-type posteriors through the couplings,
and displacement interpolation for held-out time points.
"""
import os

os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")

import numpy as np


def _coupling(X0, X1, a, eps=0.05, reg_m=(1.0, 50.0), dims=30):
    import ot
    A, B = X0[:, :dims].astype(np.float64), X1[:, :dims].astype(np.float64)
    M = ot.dist(A, B)
    M /= np.median(M)
    b = np.full(len(B), 1.0 / len(B))
    a = a / a.sum()
    try:
        G = ot.unbalanced.sinkhorn_unbalanced(a, b, M, eps, reg_m, numItermax=3000)
    except TypeError:  # older POT: scalar reg_m only
        G = ot.unbalanced.sinkhorn_unbalanced(a, b, M, eps, reg_m[0], numItermax=3000)
    return G


def wot_fate(data, clf, prog_idx, pos, neg, n_sub=6000, eps=0.05, dims=30, seed=0, times=None, condition=None):
    """Fate bias of start-time progenitors (indices into data['x'][0])."""
    from ..data.dataset import cells, time_index
    rng = np.random.default_rng(seed)
    times = times or data["times"]
    xs, gs = [], []
    for k, t in enumerate(times):
        i = time_index(data, t)
        m = np.where(cells(data, t, "all", condition if k > 0 else None))[0]
        pick = rng.choice(m, min(n_sub, len(m)), replace=False)
        if k == 0:
            pick = np.union1d(pick, prog_idx)
        xs.append(data["x"][i][pick].numpy())
        gs.append((data["growth"][i][pick].numpy(), pick))
    F = clf.predict_proba(xs[-1])
    for k in range(len(times) - 2, -1, -1):
        dt = times[k + 1] - times[k]
        a = np.exp(gs[k][0] * dt)
        G = _coupling(xs[k], xs[k + 1], a, eps, dims=dims)
        F = G @ F
        F /= np.clip(F.sum(1, keepdims=True), 1e-300, None)
    pos_i, neg_i = clf.index(pos), clf.index(neg)
    lookup = {c: r for r, c in enumerate(gs[0][1])}
    rows = np.array([lookup[c] for c in prog_idx])
    p, n = F[rows, pos_i], F[rows, neg_i]
    return p / np.clip(p + n, 1e-12, None)


def wot_interpolate(X0, X1, a, s, n_out=4000, eps=0.05, dims=30, seed=0):
    """McCann displacement interpolation at fraction s in [0, 1] between X0 and X1."""
    rng = np.random.default_rng(seed)
    G = _coupling(X0, X1, a, eps, dims=dims)
    P = G.ravel() / G.sum()
    idx = rng.choice(P.size, n_out, p=P)
    i, j = np.unravel_index(idx, G.shape)
    return (1 - s) * X0[i] + s * X1[j]
