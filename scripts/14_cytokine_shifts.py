"""Step 14: clone-matched fate shifts between media (the counterfactual test of the cytokine experiment).

On Day 2 every progenitor sits in the same cocktail; its sisters are then re-plated into different media
(Weinreb et al. 2020, Supp Methods 8.1). A Day-2 cell that has sisters in two media A and B therefore
has an observed fate-bias shift  bias_B - bias_A  that is due to the medium alone. A model without a
medium input (PRESCIENT) must predict a shift of 0; a model whose landscape responds to the ligands
in the medium predicts a shift per cell. For each pair of media and each run this script reports

  obs_mean, pred_mean   mean observed / predicted shift over the shared progenitors
  pcc                   Pearson correlation of predicted and observed per-cell shifts
  sign_agree            fraction of cells with the same sign (cells with observed shift 0 dropped)

and per model (seed mean ± sd) in cytokine_shifts_summary.csv, plus the per-medium composition fit
(L1 distance of cell-type fractions, Day 4 and 6) in cytokine_composition_summary.csv.

    python scripts/14_cytokine_shifts.py --config cytokine.yaml --root /kaggle/working
"""
import _common
import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon

from cctempo.data.dataset import load_data
from cctempo.runner import P
from cctempo.utils import load_config

p = _common.parser(__doc__, "cytokine.yaml")
p.add_argument("--min-shared", type=int, default=15)
args = p.parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
res_dir = P(cfg, "results", args.root)
os.makedirs(res_dir, exist_ok=True)
cl = load_data(P(cfg, "data_pt", args.root))["eval"]["clonal"]

preds = {}
for f in sorted(glob.glob(os.path.join(P(cfg, "runs", args.root), "*", "preds_best.npz"))):
    preds[os.path.basename(os.path.dirname(f))] = dict(np.load(f))

rows = []
for a, b in itertools.combinations(sorted(cl), 2):
    ia, ib = np.asarray(cl[a]["prog_idx"]), np.asarray(cl[b]["prog_idx"])
    common = np.intersect1d(ia, ib)
    if len(common) < args.min_shared:
        continue
    pa = {c: i for i, c in enumerate(ia)}
    pb = {c: i for i, c in enumerate(ib)}
    ja = np.array([pa[c] for c in common])
    jb = np.array([pb[c] for c in common])
    obs = np.asarray(cl[b]["bias"])[jb] - np.asarray(cl[a]["bias"])[ja]
    p_obs = float(wilcoxon(obs).pvalue) if np.any(obs != 0) else np.nan
    for run, z in preds.items():
        if f"mc_{a}" not in z or f"mc_{b}" not in z:
            continue
        pr = z[f"mc_{b}"][jb] - z[f"mc_{a}"][ja]
        ok = np.isfinite(pr)
        nz = ok & (obs != 0)
        pcc = float(pearsonr(pr[ok], obs[ok])[0]) if ok.sum() > 2 and pr[ok].std() > 1e-9 else np.nan
        rows.append({"medium_a": a, "medium_b": b, "n_shared": len(common), "obs_mean": float(obs.mean()),
                     "obs_wilcoxon_p": p_obs, "run": run, "model": run.rsplit("_s", 1)[0],
                     "pred_mean": float(np.nanmean(pr)), "pcc": pcc,
                     "sign_agree": float(np.mean(np.sign(pr[nz]) == np.sign(obs[nz]))) if nz.any() else np.nan})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(res_dir, "cytokine_shifts_runs.csv"), index=False)
if len(df):
    summ = (df.groupby(["medium_a", "medium_b", "n_shared", "obs_mean", "obs_wilcoxon_p", "model"])
            [["pred_mean", "pcc", "sign_agree"]].agg(["mean", "std", "count"]).round(4))
    summ.to_csv(os.path.join(res_dir, "cytokine_shifts_summary.csv"))
    print(summ.to_string())

# per-medium composition fit (only models trained on that medium are meaningful; loco runs flagged)
rl = os.path.join(res_dir, "runs_long.csv")
if os.path.exists(rl):
    r = pd.read_csv(rl)
    cols = [c for c in r.columns if c.startswith("l1_t") or c.startswith("w2_t") or c.startswith("kl_bits_t")]
    r["held_out_medium"] = [c not in str(tc).split(",") if isinstance(tc, str) and tc else False
                            for c, tc in zip(r["condition"], r["train_conditions"])]
    comp = r.groupby(["condition", "model", "held_out_medium"])[cols].agg(["mean", "std"]).round(3)
    comp.to_csv(os.path.join(res_dir, "cytokine_composition_summary.csv"))
    print(comp.to_string())
