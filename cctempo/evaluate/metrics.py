"""Evaluation indexes (answers Reviewer 1: 'more evaluation indexes are required')."""
import os

os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def fate_metrics(pred, true, mask=None):
    """Clonal fate-bias agreement between predictions and lineage-tracing ground truth."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    out = {"n": int(len(true)), "mean_pred": float(pred.mean()), "mean_true": float(true.mean()),
           "mean_abs_err": float(abs(pred.mean() - true.mean()))}
    out["pcc"] = float(pearsonr(pred, true)[0]) if np.std(pred) > 0 else float("nan")
    out["spearman"] = float(spearmanr(pred, true)[0]) if np.std(pred) > 0 else float("nan")
    if mask is not None and mask.sum() > 2 and np.std(pred[mask]) > 0:
        out["pcc_masked"] = float(pearsonr(pred[mask], true[mask])[0])
        out["n_masked"] = int(mask.sum())
    sel = true != 0.5
    yb = (true[sel] > 0.5).astype(int)
    if sel.sum() > 2 and 0 < yb.sum() < len(yb):
        out["auroc"] = float(roc_auc_score(yb, pred[sel]))
        out["auprc"] = float(average_precision_score(yb, pred[sel]))
    out["mae"] = float(np.mean(np.abs(pred - true)))
    out["rmse"] = float(np.sqrt(np.mean((pred - true) ** 2)))
    # the paper's "clonal fate deviation": |pred - true| / true
    out["fate_deviation_median"] = float(np.median(np.abs(pred - true) / np.clip(true, 1e-3, None)))
    return out


def _sub(x, n, rng):
    x = np.asarray(x)
    return x if x.shape[0] <= n else x[rng.choice(x.shape[0], n, replace=False)]


def wasserstein(x, y, n=3000, seed=0, p=2):
    """Exact W_p between subsamples (POT)."""
    import ot
    rng = np.random.default_rng(seed)
    x, y = _sub(x, n, rng), _sub(y, n, rng)
    M = ot.dist(x, y, metric="sqeuclidean" if p == 2 else "euclidean")
    val = ot.emd2(np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y)), M, numItermax=2_000_000)
    return float(np.sqrt(val)) if p == 2 else float(val)


def mmd_rbf(x, y, n=3000, seed=0):
    """Unbiased MMD^2 with a sum of RBF kernels at multiples of the median distance."""
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(_sub(x, n, rng), dtype=torch.float32)
    y = torch.as_tensor(_sub(y, n, rng), dtype=torch.float32)
    xy = torch.cat([x, y])
    d2 = torch.cdist(xy, xy).pow(2)
    med = d2[d2 > 0].median()
    K = sum(torch.exp(-d2 / (s * med)) for s in (0.25, 0.5, 1.0, 2.0, 4.0))
    nx = x.shape[0]
    kxx, kyy, kxy = K[:nx, :nx], K[nx:, nx:], K[:nx, nx:]
    m, k = nx, y.shape[0]
    val = ((kxx.sum() - kxx.diag().sum()) / (m * (m - 1)) + (kyy.sum() - kyy.diag().sum()) / (k * (k - 1))
           - 2 * kxy.mean())
    return float(val)


def energy_distance(x, y, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(_sub(x, n, rng), dtype=torch.float32)
    y = torch.as_tensor(_sub(y, n, rng), dtype=torch.float32)
    return float(2 * torch.cdist(x, y).mean() - torch.cdist(x, x).mean() - torch.cdist(y, y).mean())


def distribution_metrics(x_sim, x_obs, seed=0, n=3000):
    return {"w2": wasserstein(x_sim, x_obs, n, seed), "mmd": mmd_rbf(x_sim, x_obs, n, seed),
            "energy": energy_distance(x_sim, x_obs, n, seed)}


def composition_metrics(pred_frac: dict, true_frac: dict):
    keys = sorted(set(pred_frac) | set(true_frac))
    p = np.array([pred_frac.get(k, 0.0) for k in keys]) + 1e-12
    q = np.array([true_frac.get(k, 0.0) for k in keys]) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    return {"l1": float(np.abs(p - q).sum()), "js": float(jensenshannon(p, q, base=2) ** 2)}


def observed_composition(labels):
    labels = np.asarray(labels).astype(str)
    u, c = np.unique(labels, return_counts=True)
    return dict(zip(u.tolist(), (c / c.sum()).tolist()))
