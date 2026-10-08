"""Step 1: download Weinreb et al. 2020 (in vitro or cytokine perturbation), log-normalise,
compute lineage-tracing ground truth, and save an .h5ad (+ eval targets).

    python scripts/01_prepare_weinreb.py --config weinreb.yaml  --root /kaggle/working
    python scripts/01_prepare_weinreb.py --config cytokine.yaml --root /kaggle/working
"""
import _common
import os
import pickle

import numpy as np

from cctempo.data.preprocess import normalize_log
from cctempo.data.weinreb import clonal_fate_bias, load_anndata
from cctempo.runner import P
from cctempo.utils import load_config, save_json

p = _common.parser(__doc__, "weinreb.yaml")
p.add_argument("--force", action="store_true", help="redo even if the outputs already exist")
args = p.parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
pc = cfg["prepare"]
os.makedirs(P(cfg, "work", args.root), exist_ok=True)
if not args.force and os.path.exists(P(cfg, "h5ad", args.root)) and         os.path.exists(os.path.join(P(cfg, "work", args.root), "clonal_targets.pkl")):
    print("[skip] normalised h5ad and clonal targets already exist (use --force to redo)")
    raise SystemExit(0)

adata = load_anndata(pc["kind"], P(cfg, "raw", args.root))
print(adata)
tk, ck = pc["time_key"], pc["celltype_key"]
adata.obs[tk] = adata.obs[tk].astype(float)
normalize_log(adata, already_normalized=True)

ev = cfg["eval"]
clones = adata.obsm.pop("clones")
targets = {}
if pc.get("condition_key"):
    for cond in ev["conditions"]:
        prog, bias, npos, nneg = clonal_fate_bias(adata.obs, clones, tk, ck, ev["start_time"], ev["pos"], ev["neg"],
                                                  ev["undiff"], pc["condition_key"], cond)
        targets[cond] = {"rows": prog, "bias": bias, "n_pos": npos, "n_neg": nneg}
        print(f"[targets] {cond}: {len(prog)} progenitors, mean bias {bias.mean():.3f}")
else:
    prog, bias, npos, nneg = clonal_fate_bias(adata.obs, clones, tk, ck, ev["start_time"], ev["pos"], ev["neg"], ev["undiff"])
    targets[None] = {"rows": prog, "bias": bias, "n_pos": npos, "n_neg": nneg}
    print(f"[targets] {len(prog)} progenitors, mean bias {bias.mean():.3f}")

with open(os.path.join(P(cfg, "work", args.root), "clonal_targets.pkl"), "wb") as f:
    pickle.dump(targets, f)
import scipy.sparse as sp  # noqa: E402
sp.save_npz(os.path.join(P(cfg, "work", args.root), "clones.npz"), clones.tocsr())
adata.write_h5ad(P(cfg, "h5ad", args.root), compression="gzip")
save_json({"n_cells": adata.n_obs, "n_genes": adata.n_vars,
           "times": adata.obs[tk].value_counts().to_dict(),
           "types": adata.obs[ck].value_counts().to_dict(),
           "n_progenitors": {str(k): int(len(v["rows"])) for k, v in targets.items()}},
          os.path.join(P(cfg, "work", args.root), "prepare_summary.json"))
print("done")
