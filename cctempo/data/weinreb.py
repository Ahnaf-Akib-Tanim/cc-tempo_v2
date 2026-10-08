"""Weinreb et al. 2020 (Science) LARRY lineage-tracing data: in vitro and cytokine perturbation.

Source: https://github.com/AllonKleinLab/paper-data (files hosted on kleintools.hms.harvard.edu)
"""
import gzip
import os
import shutil
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE = "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/"
KINDS = {"invitro": "stateFate_inVitro", "cytokine": "stateFate_cytokinePerturbation", "invivo": "stateFate_inVivo"}
PARTS = ["normed_counts.mtx.gz", "gene_names.txt.gz", "metadata.txt.gz", "clone_matrix.mtx.gz"]


def download(kind, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for part in PARTS:
        fn = f"{KINDS[kind]}_{part}"
        dst = os.path.join(out_dir, fn)
        if not os.path.exists(dst):
            print(f"[download] {fn}")
            with urllib.request.urlopen(BASE + fn) as r, open(dst + ".part", "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
            os.replace(dst + ".part", dst)
        paths[part.split(".")[0]] = dst
    return paths


def read_mtx(path, chunksize=20_000_000):
    """Fast MatrixMarket reader (coordinate format, gz) using pandas' C parser."""
    with gzip.open(path, "rt") as f:
        header = []
        for line in f:
            if line.startswith("%"):
                continue
            header = line.split()
            break
    n_rows, n_cols, nnz = map(int, header[:3])
    rows, cols, vals = [], [], []
    reader = pd.read_csv(path, sep=r"\s+", comment="%", header=None, skiprows=1, chunksize=chunksize,
                         dtype={0: np.int32, 1: np.int32, 2: np.float32}, engine="c")
    first = True
    for chunk in reader:
        if first:  # the size line is the first non-comment row
            chunk = chunk.iloc[1:] if chunk.iloc[0, 0] == n_rows and chunk.iloc[0, 1] == n_cols else chunk
            first = False
        rows.append(chunk[0].values - 1)
        cols.append(chunk[1].values - 1)
        vals.append(chunk[2].values if chunk.shape[1] > 2 else np.ones(len(chunk), np.float32))
    M = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_rows, n_cols))
    return M.tocsr()


def load_anndata(kind, raw_dir):
    import anndata as ad
    p = download(kind, raw_dir)
    X = read_mtx(p["normed_counts"])
    genes = pd.read_csv(p["gene_names"], header=None)[0].astype(str).values
    meta = pd.read_csv(p["metadata"], sep="\t")
    clones = orient_clones(read_mtx(p["clone_matrix"]), len(meta))
    adata = ad.AnnData(X=X.astype(np.float32), obs=meta.reset_index(drop=True))
    adata.var_names = pd.Index(genes)
    adata.var_names_make_unique()
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    adata.obsm["clones"] = clones
    return adata


def orient_clones(C, n_cells):
    """Clone matrices are cells x clones (in vitro) or clones x cells (cytokine release)."""
    if C.shape[0] != n_cells and C.shape[1] == n_cells:
        C = C.T.tocsr()
    assert C.shape[0] == n_cells, (C.shape, n_cells)
    return C


def clonal_fate_bias(obs, clones, time_key="Time point", type_key="Cell type annotation", start=2.0,
                     pos="Neutrophil", neg="Monocyte", undiff="Undifferentiated", condition_key=None,
                     condition=None):
    """Original CC-Tempo / PRESCIENT benchmark: start-time undifferentiated cells whose clone has at
    least one pos/neg cell later; bias = (n_pos + 1) / (n_pos + n_neg + 2).

    With condition (cytokine data) only later cells grown in that condition are counted, giving a
    per-condition, per-clone ground truth of how the medium (i.e. the ligands) shifts fate.
    Returns (row indices of progenitors in obs, bias array, n_pos, n_neg).
    """
    t = obs[time_key].astype(float).values
    ty = obs[type_key].astype(str).values
    later = t > start
    if condition is not None:
        later &= obs[condition_key].astype(str).values == condition
    is_pos = sp.diags((later & (ty == pos)).astype(np.float32))
    is_neg = sp.diags((later & (ty == neg)).astype(np.float32))
    C = orient_clones(sp.csr_matrix(clones), len(obs))
    pos_per_clone = np.asarray((is_pos @ C).sum(0)).ravel()
    neg_per_clone = np.asarray((is_neg @ C).sum(0)).ravel()
    has = (pos_per_clone + neg_per_clone) > 0
    in_good_clone = np.asarray(C[:, has].sum(1)).ravel() > 0
    prog = np.where((t == start) & (ty == undiff) & in_good_clone)[0]
    npos = np.asarray(C[prog] @ pos_per_clone).ravel()
    nneg = np.asarray(C[prog] @ neg_per_clone).ravel()
    bias = (npos + 1) / (npos + nneg + 2)
    return prog, bias, npos, nneg
