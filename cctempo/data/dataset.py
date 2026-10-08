"""The `data.pt` container shared by every model, script and the web app.

Keys
----
x          list[Tensor (n_t, d)]  PCA coordinates per time point
times      list[float]
growth     list[Tensor (n_t,)]   per-cell net growth rate (birth - death)
celltype   list[np.ndarray[str]]
condition  list[np.ndarray[str]] or None
split      list[np.ndarray[bool]] True = train, False = validation
obs_index  list[np.ndarray[int]] row index of each cell in the processed AnnData
genes      list[str]             genes used for PCA (HVGs + forced LR/TF genes)
pca        {components (d, G), mean (G,)}
scaler     {mean (G,), scale (G,)}
lr         {ligand, receptor, lig_genes, rec_genes, prior, family, exogenous} or None
v1         list[Tensor (n_t, F)] static sending/receiving features for the v1 model, or None
meta       dict (dataset name, preprocessing parameters, LR method, ...)
"""
import numpy as np
import torch


def save_data(data: dict, path: str) -> None:
    torch.save(data, path)


def load_data(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    data["x"] = [torch.as_tensor(x).float() for x in data["x"]]
    data["growth"] = [torch.as_tensor(g).float() for g in data["growth"]]
    if data.get("v1") is not None:
        data["v1"] = [torch.as_tensor(v).float() for v in data["v1"]]
    return data


def time_index(data: dict, t: float) -> int:
    for i, tt in enumerate(data["times"]):
        if abs(float(tt) - float(t)) < 1e-6:
            return i
    raise KeyError(f"time {t} not in {data['times']}")


def cells(data: dict, t: float, split: str = "all", condition=None):
    """Boolean mask of cells at time t for a split ('train'|'val'|'all') and optional condition."""
    i = time_index(data, t)
    n = data["x"][i].shape[0]
    m = np.ones(n, dtype=bool)
    if split == "train":
        m &= data["split"][i]
    elif split == "val":
        m &= ~data["split"][i]
    if condition is not None and data.get("condition") is not None:
        cond = data["condition"][i]
        if isinstance(condition, (list, tuple, set)):
            m &= np.isin(cond, list(condition))
        else:
            m &= cond == condition
    return m


def make_split(n: int, val_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    split = np.ones(n, dtype=bool)
    split[rng.choice(n, size=int(round(val_frac * n)), replace=False)] = False
    return split
