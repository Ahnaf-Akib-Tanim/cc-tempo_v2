"""Scanpy preprocessing shared by all datasets and construction of the data.pt container."""
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA

from ..growth import growth_rates
from .dataset import make_split
from .genesets import APOPTOSIS, CELL_CYCLE


def normalize_log(adata, target_sum=1e4, already_normalized=False):
    import scanpy as sc
    # raw counts and already L1-normalised matrices (e.g. Weinreb 'normed_counts') are both
    # brought to `target_sum` per cell before log1p
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_genes(adata, n_hvg=2500, force_genes=(), min_cells=10, batch_key=None):
    import scanpy as sc
    tmp = adata.copy()
    sc.pp.filter_genes(tmp, min_cells=min_cells)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat", batch_key=batch_key)
    hv = set(tmp.var_names[tmp.var["highly_variable"]])
    forced = [g for g in force_genes if g in adata.var_names]
    genes = [g for g in adata.var_names if g in hv or g in set(forced)]
    return genes


def _dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def build_data(adata, genes, time_key, celltype_key, condition_key=None, n_pcs=50, val_frac=0.1, seed=0,
               lr=None, v1=None, meta=None, max_scale=10.0, growth_kwargs=None, pca_fit_max=60000):
    """adata must be log-normalised (all genes, used for growth scores); `genes` = model genes."""
    X = _dense(adata[:, genes].X).astype(np.float32)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.clip((X - mu) / sd, -max_scale, max_scale)
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(Xs.shape[0], min(pca_fit_max, Xs.shape[0]), replace=False)
    pca = PCA(n_components=n_pcs, random_state=seed).fit(Xs[fit_idx])
    Z = pca.transform(Xs).astype(np.float32)

    # growth from the full gene space (cell-cycle / apoptosis genes need not be HVGs)
    allg = list(adata.var_names)
    need = [g for g in CELL_CYCLE + APOPTOSIS if g in set(allg)]
    Xg = _dense(adata[:, need].X)
    gr = growth_rates(Xg, need, Z[:, :min(30, n_pcs)], CELL_CYCLE, APOPTOSIS, **(growth_kwargs or {}))

    times = sorted(adata.obs[time_key].astype(float).unique())
    tcol = adata.obs[time_key].astype(float).values
    data = {"x": [], "times": [float(t) for t in times], "growth": [], "celltype": [], "split": [],
            "obs_index": [], "condition": [] if condition_key else None, "v1": [] if v1 is not None else None}
    for t in times:
        idx = np.where(tcol == t)[0]
        data["x"].append(torch.from_numpy(Z[idx]))
        data["growth"].append(torch.from_numpy(gr["growth"][idx].astype(np.float32)))
        data["celltype"].append(adata.obs[celltype_key].astype(str).values[idx])
        data["split"].append(make_split(len(idx), val_frac, seed + int(t * 10)))
        data["obs_index"].append(idx)
        if condition_key:
            data["condition"].append(adata.obs[condition_key].astype(str).values[idx])
        if v1 is not None:
            data["v1"].append(torch.from_numpy(np.asarray(v1)[idx].astype(np.float32)))
    data["genes"] = list(genes)
    data["pca"] = {"components": pca.components_.astype(np.float32), "mean": pca.mean_.astype(np.float32),
                   "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32)}
    data["scaler"] = {"mean": mu.astype(np.float32), "scale": sd.astype(np.float32)}
    data["lr"] = lr
    data["meta"] = dict(meta or {}, n_pcs=n_pcs, n_genes=len(genes), growth_genes=(gr["n_birth_genes"], gr["n_death_genes"]))
    return data
