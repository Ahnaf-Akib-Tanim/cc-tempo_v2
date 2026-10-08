"""Step 2: cell-cell communication with LIANA (consensus + individual methods), per time point.

Outputs (in paths.work):
  liana_table.csv.gz       long table of all interactions/methods/time points
  lr_<method>.json         selected LR pairs (top-K) per method -> data variants for robustness
  liana_method_overlap.csv Jaccard overlap of top-K pairs between methods (Reviewer 1(c))

    python scripts/02_liana_ccc.py --config weinreb.yaml --root /kaggle/working
"""
import _common
import json
import os

import anndata as ad

from cctempo.ccc.liana_prior import method_overlap, run_liana, select_pairs
from cctempo.runner import P
from cctempo.utils import load_config

args = _common.parser(__doc__, "weinreb.yaml").parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
pc = cfg["prepare"]
adata = ad.read_h5ad(P(cfg, "h5ad", args.root))
tk = pc["time_key"] if pc["time_key"] != "auto" else adata.uns.get("cct_time_key")
ck = pc["celltype_key"] if pc["celltype_key"] != "auto" else adata.uns.get("cct_celltype_key")

out_tab = P(cfg, "liana", args.root)
if os.path.exists(out_tab):
    import pandas as pd
    table = pd.read_csv(out_tab)
    print(f"[liana] reusing {out_tab}")
else:
    table = run_liana(adata, groupby=ck, time_key=tk, methods=pc["liana_methods"], resource_name=pc["liana_resource"],
                      max_cells_per_group=pc.get("liana_max_cells_per_group", 1500))
    table.to_csv(out_tab, index=False)

work = P(cfg, "work", args.root)
for m in table["method"].unique():
    lr = select_pairs(table, adata.var_names, method=m, top_k=pc["lr_top_k"], force=pc.get("force_cytokines"))
    with open(os.path.join(work, f"lr_{m}.json"), "w") as f:
        json.dump(lr, f, indent=1)
    print(f"[select] {m}: {len(lr['ligand'])} pairs; top: {list(zip(lr['ligand'], lr['receptor']))[:8]}")
if table["method"].nunique() > 1:
    method_overlap(table, pc["lr_top_k"]).to_csv(os.path.join(work, "liana_method_overlap.csv"))
print("done")
