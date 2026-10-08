"""Proliferation weights (birth - death), following PRESCIENT with a corrected smoothing step.

Reviewer 2 asked (Q4) why the mean of z-scores is used and (Q5) whether Eq. 11 is right.
* Score: s_i = mean_g z_ig over genes g of a set (cell cycle / apoptosis). Averaging z-scores
  gives every gene equal weight regardless of its expression level (otherwise a few highly
  expressed genes dominate) -- the same idea as Seurat AddModuleScore / scanpy score_genes. We
  also provide 'scanpy' scoring (control-gene corrected) for a robustness check.
* Smoothing (Eq. 11, corrected): personalised graph diffusion on a kNN graph
      s^(k+1) = (1 - a) s^(0) + a A s^(k),   A = row-normalised kNN adjacency, 0 < a < 1,
  which converges to s* = (1 - a)(I - aA)^(-1) s^(0) (spectral radius of aA < 1).
* Rates: logistic maps of the smoothed scores, b(s) = b_min + (b_max - b_min) / (1 + exp(-4 (s - c) / w)).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def zscore_score(X, gene_idx):
    """Mean z-score over genes; X is (cells, genes) log-expression (dense)."""
    sub = np.asarray(X[:, gene_idx], dtype=np.float64)
    mu, sd = sub.mean(0), sub.std(0)
    sd[sd == 0] = 1.0
    return ((sub - mu) / sd).mean(1)


def knn_smooth(score, emb, k=15, a=0.5, n_iter=20):
    nn = NearestNeighbors(n_neighbors=k).fit(emb)
    _, ind = nn.kneighbors(emb)
    s0 = np.asarray(score, float)
    s = s0.copy()
    for _ in range(n_iter):
        s = (1 - a) * s0 + a * s[ind].mean(1)
    return s


def logistic_rate(s, r_min, r_max, center, width):
    return r_min + (r_max - r_min) / (1.0 + np.exp(-4.0 * (s - center) / width))


def growth_rates(X, genes, emb, birth_genes, death_genes, k=15, a=0.5, n_iter=20,
                 birth=(0.3, 1.7, 0.25, 0.5), death=(0.3, 1.7, 0.1, 0.2)):
    """Return dict(birth, death, growth, birth_score, death_score). Defaults follow PRESCIENT."""
    pos = {g: i for i, g in enumerate(genes)}
    bi = [pos[g] for g in birth_genes if g in pos]
    di = [pos[g] for g in death_genes if g in pos]
    if len(bi) < 5 or len(di) < 5:
        raise ValueError(f"too few cell-cycle ({len(bi)}) / apoptosis ({len(di)}) genes in dataset")
    sb = knn_smooth(zscore_score(X, bi), emb, k, a, n_iter)
    sd = knn_smooth(zscore_score(X, di), emb, k, a, n_iter)
    b = logistic_rate(sb, birth[0], birth[1], birth[2], birth[3])
    d = logistic_rate(sd, death[0], death[1], death[2], death[3])
    return {"birth": b, "death": d, "growth": b - d, "birth_score": sb, "death_score": sd,
            "n_birth_genes": len(bi), "n_death_genes": len(di)}
