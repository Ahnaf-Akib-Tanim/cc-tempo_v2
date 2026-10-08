"""Step 3: genes (HVGs + every LR subunit from every LIANA method + TFs), growth, PCA, v1
features and evaluation targets -> data.pt.

    python scripts/03_build_dataset.py --config weinreb.yaml --root /kaggle/working
"""
import _common
import glob
import json
import os
import pickle

import anndata as ad
import numpy as np

from cctempo.ccc.liana_prior import v1_features
from cctempo.data.dataset import save_data
from cctempo.data.genesets import TFS, V1_LINEAGE_GENES
from cctempo.data.preprocess import build_data, select_genes
from cctempo.runner import P
from cctempo.utils import load_config

args = _common.parser(__doc__, "weinreb.yaml").parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
pc = cfg["prepare"]
work = P(cfg, "work", args.root)
adata = ad.read_h5ad(P(cfg, "h5ad", args.root))
tk = pc["time_key"] if pc["time_key"] != "auto" else adata.uns["cct_time_key"]
ck = pc["celltype_key"] if pc["celltype_key"] != "auto" else adata.uns["cct_celltype_key"]
cond_key = pc.get("condition_key")

lrs = {}
for f in sorted(glob.glob(os.path.join(work, "lr_*.json"))):
    with open(f) as fh:
        lr = json.load(fh)
    lrs[lr["method"]] = lr
lr_main = lrs[pc["lr_method"]]
force = {g for lr in lrs.values() for grp in lr["lig_genes"] + lr["rec_genes"] for g in grp}
force |= {g for v in TFS.values() for g in v} | {g for v in V1_LINEAGE_GENES.values() for g in v}
force |= set(pc.get("pnas_genes", []))
genes = select_genes(adata, pc["n_hvg"], sorted(force))
print(f"[genes] {len(genes)} model genes ({len(force & set(genes))} forced)")

v1 = None
import pandas as pd  # noqa: E402
table = pd.read_csv(P(cfg, "liana", args.root))
v1, v1_names = v1_features(adata, lr_main, table, ck, tk, pc["lr_method"])
print(f"[v1] static communication features: {v1.shape}")

data = build_data(adata, genes, tk, ck, cond_key, pc["n_pcs"], pc["val_frac"], seed=0, lr=lr_main, v1=v1,
                  meta={"name": cfg["name"], "time_key": tk, "celltype_key": ck, "v1_names": v1_names})

# lineage-tracing targets -> indices into data['x'][start]
tpath = os.path.join(work, "clonal_targets.pkl")
if os.path.exists(tpath):
    with open(tpath, "rb") as f:
        targets = pickle.load(f)
    t0 = float(cfg["eval"]["start_time"])
    i0 = data["times"].index(t0)
    pos = {r: j for j, r in enumerate(data["obs_index"][i0])}
    clonal = {}
    for cond, d in targets.items():
        keep = [k for k, r in enumerate(d["rows"]) if r in pos]
        entry = {"prog_idx": np.array([pos[d["rows"][k]] for k in keep]), "bias": d["bias"][keep]}
        if cond is None:
            clonal = entry
        else:
            clonal[cond] = entry
    data["eval"] = {"clonal": clonal}
save_data(data, P(cfg, "data_pt", args.root))
print(f"[saved] {P(cfg, 'data_pt', args.root)}; times {data['times']}; cells {[x.shape[0] for x in data['x']]}")
