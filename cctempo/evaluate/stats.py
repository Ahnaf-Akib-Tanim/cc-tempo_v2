"""Statistics that work with ~5 seeds instead of 100 trainings.

* seed-level paired Wilcoxon / unpaired Mann-Whitney on per-seed metrics
* progenitor-level paired bootstrap of the PCC difference between two (ensemble) predictions:
  thousands of resamples of the ~1-3k progenitors give tight CIs even with few seeds
* Holm correction across comparisons, Cliff's delta effect size
"""
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, wilcoxon


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def seed_test(a, b, paired=True, alternative="greater"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "sd_a": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
           "sd_b": float(b.std(ddof=1)) if len(b) > 1 else 0.0, "cliffs_delta": cliffs_delta(a, b)}
    try:
        if paired and len(a) == len(b):
            out["p_wilcoxon"] = float(wilcoxon(a, b, alternative=alternative).pvalue)
        out["p_mannwhitney"] = float(mannwhitneyu(a, b, alternative=alternative).pvalue)
    except ValueError:
        pass
    return out


def bootstrap_pcc_diff(pred_a, pred_b, true, n_boot=5000, seed=0):
    """Paired bootstrap over progenitors of PCC(a) - PCC(b). Returns diff, 95% CI and one-sided p."""
    pred_a, pred_b, true = map(lambda v: np.asarray(v, float), (pred_a, pred_b, true))
    rng = np.random.default_rng(seed)
    n = len(true)
    d0 = pearsonr(pred_a, true)[0] - pearsonr(pred_b, true)[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, n, n)
        ta = true[j]
        diffs[i] = np.corrcoef(pred_a[j], ta)[0, 1] - np.corrcoef(pred_b[j], ta)[0, 1]
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    p = float((np.sum(diffs <= 0) + 1) / (n_boot + 1))
    return {"delta_pcc": float(d0), "ci_low": float(lo), "ci_high": float(hi), "p_one_sided": p}
