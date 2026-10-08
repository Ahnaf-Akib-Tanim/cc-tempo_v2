"""Step 4: synthetic ground-truth benchmark (cheap; runs on Kaggle CPU or GPU).

For each true coupling beta in {1.5 (communication drives fate), 0 (autonomous control)}:
generate data with exogenous-ligand doses 0 / 0.5 / 1, train every model on doses {0, 1} and
evaluate on all doses (0.5 = held-out signal level). Questions answered:
  Q1 does a communication-coupled model recover the dose-dependent fate shift that autonomous
     models (PRESCIENT, CC-off) cannot?                       -> composition / fate PCC at e0.5
  Q2 does it find the right channel? -> knockout effect of the TRUE pair vs DECOY pairs
  Q3 does it stay silent when beta = 0?  -> knockout effects ~ 0, no gain over CC-off

    python scripts/04_synthetic_benchmark.py --root /kaggle/working            # full
    python scripts/04_synthetic_benchmark.py --quick                           # 2-minute check
"""
import _common
import os

import numpy as np
import pandas as pd
import torch

from cctempo.data.dataset import save_data
from cctempo.data.synthetic import make_synthetic
from cctempo.interpret.perturb import FateRunner, lr_knockout_scan
from cctempo.models import load_run_model
from cctempo.runner import classifier_for, evaluate_run, load_dataset, make_cfg, train_one
from cctempo.utils import get_device, save_json

p = _common.parser(__doc__, "synthetic.yaml")
p.add_argument("--quick", action="store_true")
args = p.parse_args()
cfgp = _common.cfg_path(args.config)
base = make_cfg(cfgp, None, args.set)
sc = base["synthetic"]
seeds, models = sc["seeds"], sc["models"]
over = list(args.set)
if args.quick:
    seeds, models = [0], ["prescient", "v2_meanfield"]
    over += ["train.epochs=30", "train.eval_every=10", "eval.n_copies=30"]
dev = str(get_device(args.device))
work = os.path.join(args.root, base["paths"]["work"])
os.makedirs(work, exist_ok=True)
rows, ko_rows = [], []
for beta in sc["betas"]:
    dpath = os.path.join(work, f"data_beta{beta}.pt")
    if not os.path.exists(dpath):
        save_data(make_synthetic(beta=beta, doses=tuple(sc["doses"]), n_cells=sc["n_cells"] if not args.quick else 600), dpath)
    tag_over = over + [f"paths.data_pt={base['paths']['work']}/data_beta{beta}.pt",
                       f"paths.runs={base['paths']['work']}/runs_c_beta{beta}"]
    for preset in models:
        for s in seeds:
            out = train_one(cfgp, preset, s, args.root, dev, tag_over)
            res = evaluate_run(out, args.root, dev)
            for cond, m in res["fate"].items():
                row = {"beta": beta, "model": preset, "seed": s, "condition": cond, "heldout": cond == "e0.5",
                       "mc_pcc": m.get("pcc"), "mc_mae": m.get("mae"), "mean_pred": m.get("mean_pred"),
                       "mean_true": m.get("mean_true"), "fate_mean_err": m.get("mean_abs_err"),
                       "pi_pcc": res["pathint"].get(cond, {}).get("pcc")}
                tf = res["dist"].get(cond, {}).get("3.0", {})
                row.update({"w2_t3": tf.get("w2"), "comp_l1_t3": tf.get("l1")})
                rows.append(row)
            if preset in ("v2_meanfield", "v2_mf_sparse", "v2_mf_uncentered", "ctrl_shuffled"):
                cfg = make_cfg(cfgp, preset, tag_over)
                data = load_dataset(cfg, args.root)
                model, _ = load_run_model(out, data, dev)
                clf = classifier_for(cfg, data, args.root)
                gt = data["eval"]["synthetic"]["e0.0"]
                for cond, exo in (("e0.0", {}), ("e1.0", {"EXO": 1.0})):
                    e = model.exogenous_field(exo).detach()
                    runner = FateRunner(model, data, clf, gt["prog_idx"][:150], 0.0, 3.0,
                                        n_copies=50 if not args.quick else 10, n_bg=1000, pos="A", neg="B",
                                        device=dev, exog_fn=lambda t, e=e: e)
                    ko = lr_knockout_scan(runner, level="family")
                    ko["beta"], ko["seed"], ko["model"], ko["condition"] = beta, s, preset, cond
                    ko_rows.append(ko)
res_dir = os.path.join(args.root, base["paths"]["results"])
os.makedirs(res_dir, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(res_dir, "synthetic_benchmark.csv"), index=False)
if ko_rows:
    pd.concat(ko_rows).to_csv(os.path.join(res_dir, "synthetic_knockouts.csv"), index=False)
summary = df.groupby(["beta", "model", "condition"])[["fate_mean_err", "mean_pred", "mean_true", "comp_l1_t3", "w2_t3", "mc_pcc", "pi_pcc"]].mean().round(4)
print(summary.to_string())
summary.to_csv(os.path.join(res_dir, "synthetic_summary.csv"))
