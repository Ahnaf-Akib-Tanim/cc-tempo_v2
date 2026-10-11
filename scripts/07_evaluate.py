"""Step 7: evaluate every finished run of a dataset and build the comparison tables + statistics.

Outputs in paths.results:
  runs_long.csv        one row per run x condition (all metrics)
  summary.csv          mean / sd per model (preset + tag)
  stats_seed.csv       seed-level tests of each model vs references (Wilcoxon / Mann-Whitney, Holm)
  stats_bootstrap.csv  progenitor-level paired bootstrap of PCC(ensemble A) - PCC(ensemble B)

    python scripts/07_evaluate.py --config weinreb.yaml --root /kaggle/working
    python scripts/07_evaluate.py --config weinreb.yaml --root . --aggregate-only   # locally, on downloaded runs
"""
import _common
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

from cctempo.evaluate.stats import bootstrap_pcc_diff, holm, seed_test
from cctempo.runner import P, evaluate_run
from cctempo.utils import load_config

p = _common.parser(__doc__, "weinreb.yaml")
p.add_argument("--aggregate-only", action="store_true")
p.add_argument("--refs", nargs="*", default=["prescient", "ctrl_off", "ctrl_shuffled", "v1"])
p.add_argument("--metric", default="pcc")
args = p.parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
runs = sorted(d for d in glob.glob(os.path.join(P(cfg, "runs", args.root), "*")) if os.path.exists(os.path.join(d, "run_summary.json")))
print(f"{len(runs)} finished runs")
rows, preds = [], {}
for r in runs:
    mpath = os.path.join(r, "metrics_best.json")
    if not os.path.exists(mpath):
        if args.aggregate_only:
            continue
        print("[eval]", r)
        evaluate_run(r, args.root, args.device)
    with open(mpath) as f:
        m = json.load(f)
    with open(os.path.join(r, "config_used.yaml")) as f:
        used = yaml.safe_load(f)
    name = os.path.basename(r)
    tag = name.rsplit("_s", 1)[0]
    base = {"run": name, "model": tag, "preset": used.get("preset"), "seed": m.get("seed"),
            "lr_method": used.get("lr_method"), "train_conditions": ",".join(map(str, used["train"].get("conditions") or [])),
            "holdout_times": ",".join(map(str, used["train"].get("holdout_times") or []))}
    conds = set(m["fate"]) | set(m["dist"])
    for c in conds:
        row = dict(base, condition=c)
        for k, v in m["fate"].get(c, {}).items():
            row[f"mc_{k}"] = v
        for k, v in m["pathint"].get(c, {}).items():
            if k == "kl":
                for t, kv in v.items():
                    row[f"kl_bits_t{t}"] = kv["kl_bits"]
                    row[f"entropy_bits_t{t}"] = kv["entropy_bits"]
            else:
                row[f"pi_{k}"] = v
        for t, dm in m["dist"].get(c, {}).items():
            for k, v in dm.items():
                row[f"{k}_t{t}"] = v
        rows.append(row)
    pz = os.path.join(r, "preds_best.npz")
    if os.path.exists(pz):
        preds[name] = dict(np.load(pz))
out = P(cfg, "results", args.root)
os.makedirs(out, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "runs_long.csv"), index=False)
num = df.select_dtypes("number").columns.drop("seed", errors="ignore")
summ = df.groupby(["model", "condition"])[list(num)].agg(["mean", "std", "count"])
summ.to_csv(os.path.join(out, "summary.csv"))
key = f"mc_{args.metric}"
if key in df:
    show = df.groupby(["model", "condition"])[key].agg(["mean", "std", "count"]).round(4)
    print(show.to_string())

# ---- statistics
stat_rows, boot_rows = [], []
for cond, dc in df.groupby("condition"):
    if key not in dc:
        continue
    models = sorted(dc["model"].unique())
    for ref in [r for r in args.refs if r in models]:
        for mdl in models:
            if mdl == ref:
                continue
            a = dc[dc.model == mdl].set_index("seed")[key].dropna()
            b = dc[dc.model == ref].set_index("seed")[key].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            common = a.index.intersection(b.index)
            paired = len(common) >= 3
            st = seed_test(a.loc[common].values if paired else a.values, b.loc[common].values if paired else b.values, paired)
            stat_rows.append({"condition": cond, "model": mdl, "ref": ref, **st})
            # ensemble bootstrap over progenitors
            pa = [preds[r][f"mc_{cond}"] for r in dc[dc.model == mdl].run if r in preds and f"mc_{cond}" in preds[r]]
            pb = [preds[r][f"mc_{cond}"] for r in dc[dc.model == ref].run if r in preds and f"mc_{cond}" in preds[r]]
            tgt = next((preds[r][f"target_{cond}"] for r in dc.run if r in preds and f"target_{cond}" in preds[r]), None)
            if pa and pb and tgt is not None:
                boot_rows.append({"condition": cond, "model": mdl, "ref": ref, "n_a": len(pa), "n_b": len(pb),
                                  **bootstrap_pcc_diff(np.mean(pa, 0), np.mean(pb, 0), tgt)})
for rows_, fname, pcol in ((stat_rows, "stats_seed.csv", "p_mannwhitney"), (boot_rows, "stats_bootstrap.csv", "p_one_sided")):
    if rows_:
        st = pd.DataFrame(rows_)
        if pcol in st:
            st[f"{pcol}_holm"] = holm(st[pcol].fillna(1.0).values)
        st.to_csv(os.path.join(out, fname), index=False)
        print(f"\n{fname}\n", st.round(4).to_string())
