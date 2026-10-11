"""Step 13: fate-bias accuracy on early (CD34+) progenitors and the held-out-clone upper bound.

Weinreb et al. 2020 (Fig 5c,d; Supp Methods 6, 13.5) report that WOT, PBA and FateID reach R < 0.26 on
the early progenitor region, against R = 0.5 for a prediction from held-out clonal data. This script
puts our runs on the same footing, from saved predictions only (no retraining, no GPU):

  early progenitors : Day-2 progenitors whose graph-smoothed Cd34 is high (kNN graph on the Day-2
                      PCA coordinates, N=10 rounds, beta=0.1 as in Supp Methods 4.1). The paper's cut-off
                      (1 smoothed UMI) depends on its normalisation, so the most immature half and
                      quarter are reported as well.
  upper bound       : hold out 50% of clones; predict every other progenitor by the mean bias of
                      held-out progenitors among its 400 nearest Day-2 neighbours (Supp Methods 13.5);
                      20 random splits.

Outputs (results/): early_progenitors_runs.csv, early_progenitors_summary.csv,
early_progenitors_bootstrap.csv, early_progenitors_info.json

    python scripts/13_early_progenitors.py --config weinreb.yaml --root /kaggle/working
"""
import _common
import glob
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

from cctempo.data.dataset import load_data, time_index
from cctempo.evaluate.stats import bootstrap_pcc_diff, holm
from cctempo.runner import P
from cctempo.utils import load_config, save_json

p = _common.parser(__doc__, "weinreb.yaml")
p.add_argument("--knn", type=int, default=10, help="neighbours of the smoothing graph")
p.add_argument("--rounds", type=int, default=10)
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--threshold", type=float, default=1.0, help="smoothed Cd34 (normalised counts) cut-off")
p.add_argument("--splits", type=int, default=20)
p.add_argument("--refs", nargs="*", default=["prescient", "ctrl_off", "ctrl_shuffled", "ctrl_random", "v1"])
args = p.parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
res_dir = P(cfg, "results", args.root)
os.makedirs(res_dir, exist_ok=True)

data = load_data(P(cfg, "data_pt", args.root))
t0 = float(cfg["eval"]["start_time"])
i0 = time_index(data, t0)
X0 = data["x"][i0].numpy()
rows0 = np.asarray(data["obs_index"][i0])          # Day-2 cells -> rows of the processed h5ad
prog = np.asarray(data["eval"]["clonal"]["prog_idx"])
target = np.asarray(data["eval"]["clonal"]["bias"], float)

# --- graph-smoothed Cd34 on the Day-2 kNN graph -------------------------------------------------
import anndata as ad  # noqa: E402
A = ad.read_h5ad(P(cfg, "h5ad", args.root), backed="r")
gene = "Cd34"
col = list(A.var_names).index(gene)
v = A[np.sort(rows0), col].X
v = np.asarray(v.todense() if sp.issparse(v) else v).ravel()
order = np.argsort(np.argsort(rows0))              # back to the data order of rows0
cd34 = np.expm1(v[order])                          # log1p(normalised counts) -> normalised counts
nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(X0)
_, nbr = nn.kneighbors(X0)
nbr = nbr[:, 1:]
s = cd34.copy()
for _ in range(args.rounds):
    s = (1 - args.beta) * s + args.beta * s[nbr].mean(1)
s_prog = s[prog]
subsets = {"all": np.ones(len(prog), bool),
           f"cd34_gt{args.threshold:g}": s_prog > args.threshold,
           "cd34_top50": s_prog >= np.quantile(s_prog, 0.5),
           "cd34_top25": s_prog >= np.quantile(s_prog, 0.75)}
info = {"n_progenitors": int(len(prog)), "subset_sizes": {k: int(m.sum()) for k, m in subsets.items()},
        "smoothed_cd34_quantiles": dict(zip(["q10", "q25", "q50", "q75", "q90"],
                                            np.quantile(s_prog, [.1, .25, .5, .75, .9]).round(3).tolist()))}
print(info)


def pcc(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(pearsonr(a[ok], b[ok])[0]) if ok.sum() > 2 else np.nan


# --- held-out-clone upper bound (Supp Methods 13.5) ---------------------------------------------
C = sp.load_npz(os.path.join(P(cfg, "work", args.root), "clones.npz")).tocsr()
clone_of = np.asarray(C[rows0[prog]].argmax(1)).ravel()
has_clone = np.asarray(C[rows0[prog]].sum(1)).ravel() > 0
nn400 = NearestNeighbors(n_neighbors=min(400, len(X0))).fit(X0)
_, nb400 = nn400.kneighbors(X0[prog])
pos_in_prog = -np.ones(len(X0), int)
pos_in_prog[prog] = np.arange(len(prog))
rng = np.random.default_rng(0)
ub_rows = []
uclones = np.unique(clone_of[has_clone])
for k in range(args.splits):
    held = set(rng.choice(uclones, len(uclones) // 2, replace=False).tolist())
    is_held = np.array([h and (c in held) for c, h in zip(clone_of, has_clone)])
    pred = np.full(len(prog), np.nan)
    for j in np.where(~is_held)[0]:
        idx = pos_in_prog[nb400[j]]
        idx = idx[(idx >= 0)]
        idx = idx[is_held[idx]]
        if len(idx):
            pred[j] = target[idx].mean()
    for name, m in subsets.items():
        sel = m & ~is_held
        ub_rows.append({"split": k, "subset": name, "pcc": pcc(pred[sel], target[sel])})
ub = pd.DataFrame(ub_rows).groupby("subset")["pcc"].agg(["mean", "std"]).reset_index()
print("held-out-clone upper bound:\n", ub.to_string(index=False))

# --- model and baseline predictions ------------------------------------------------------------
preds = {}
for f in sorted(glob.glob(os.path.join(P(cfg, "runs", args.root), "*", "preds_best.npz"))):
    run = os.path.basename(os.path.dirname(f))
    z = np.load(f)
    if "mc_None" in z.files:
        preds[run] = z["mc_None"]
for f in sorted(glob.glob(os.path.join(res_dir, "preds_*_None_s*.npy"))):
    name = os.path.basename(f)[len("preds_"):-4].replace("_None", "")   # e.g. WOT_s0
    preds[name] = np.load(f)
rows = []
for run, pr in preds.items():
    if len(pr) != len(prog):
        continue
    model = run.rsplit("_s", 1)[0]
    for name, m in subsets.items():
        rows.append({"run": run, "model": model, "subset": name, "n": int(m.sum()), "pcc": pcc(pr[m], target[m])})
runs_df = pd.DataFrame(rows)
summ = runs_df.groupby(["subset", "model"])["pcc"].agg(["mean", "std", "count"]).reset_index()
ubm = ub.rename(columns={"mean": "mean", "std": "std"}).assign(model="held-out clones (upper bound)", count=args.splits)
summ = pd.concat([summ, ubm[["subset", "model", "mean", "std", "count"]]], ignore_index=True)
summ = summ.sort_values(["subset", "mean"], ascending=[True, False])
print(summ.round(4).to_string(index=False))

# paired bootstrap on seed-averaged predictions: v2_meanfield vs each reference, per subset
ens = {m: np.nanmean(np.stack([preds[r] for r in runs_df[runs_df.model == m].run.unique()]), 0)
       for m in runs_df.model.unique()}
boot = []
if "v2_meanfield" in ens:
    for name, m in subsets.items():
        for ref in [r for r in args.refs if r in ens]:
            a, b, t = ens["v2_meanfield"][m], ens[ref][m], target[m]
            ok = np.isfinite(a) & np.isfinite(b)
            out = bootstrap_pcc_diff(a[ok], b[ok], t[ok])
            boot.append({"subset": name, "model": "v2_meanfield", "ref": ref, **out})
    boot = pd.DataFrame(boot)
    if len(boot):
        pcol = [c for c in boot.columns if c.startswith("p")][0]
        boot["p_holm"] = np.concatenate([holm(g[pcol].values) for _, g in boot.groupby("subset", sort=False)])
        print(boot.round(4).to_string(index=False))
        boot.to_csv(os.path.join(res_dir, "early_progenitors_bootstrap.csv"), index=False)

runs_df.to_csv(os.path.join(res_dir, "early_progenitors_runs.csv"), index=False)
summ.to_csv(os.path.join(res_dir, "early_progenitors_summary.csv"), index=False)
save_json({**info, "upper_bound": ub.to_dict("records")}, os.path.join(res_dir, "early_progenitors_info.json"))
