"""Step 6: run a tier of the experiment plan (configs/experiments.yaml); finished runs are skipped.

    python scripts/06_run_grid.py --dataset weinreb --tier tier1 --root /kaggle/working
    # two GPUs: split the list between two processes
    python scripts/06_run_grid.py --dataset weinreb --tier tier1 --shard 0/2 --device cuda:0 &
    python scripts/06_run_grid.py --dataset weinreb --tier tier1 --shard 1/2 --device cuda:1 &
    python scripts/06_run_grid.py --dataset weinreb --tier tier1 --list     # just print the plan
"""
import _common
import os
import time
import traceback

import yaml

from cctempo.data.dataset import load_data
from cctempo.runner import P, evaluate_run, make_cfg, run_name, train_one

p = _common.parser(__doc__)
p.add_argument("--dataset", required=True)
p.add_argument("--tier", default="tier1")
p.add_argument("--shard", default="0/1")
p.add_argument("--list", action="store_true")
p.add_argument("--no-eval", action="store_true")
args = p.parse_args()
with open(os.path.join(_common.PROJECT, "configs", "experiments.yaml")) as f:
    plan = yaml.safe_load(f)[args.dataset]
cfgp = _common.cfg_path(plan["config"])
jobs = []
for e in plan[args.tier]:
    for s in e["seeds"]:
        jobs.append((e["preset"], s, e.get("tag", ""), e.get("lr_method"), list(e.get("set", []))))
k, n = map(int, args.shard.split("/"))
jobs = jobs[k::n]
last_time = None
for preset, seed, tag, lrm, sets in jobs:
    if any("LAST" in x for x in sets):
        if last_time is None:
            last_time = load_data(P(make_cfg(cfgp), "data_pt", args.root))["times"][-1]
        sets = [x.replace("LAST", str(last_time)) for x in sets]
    name = run_name(preset, seed, tag)
    if args.list:
        print(name, lrm or "", sets)
        continue
    dl = os.environ.get("CCT_DEADLINE")
    if dl and float(dl) - time.time() < 20 * 60:
        print(f"[deadline] not starting {name}; run this tier again in the next session")
        continue
    try:
        out = train_one(cfgp, preset, seed, args.root, args.device, args.set + sets, tag, lrm)
        finished = os.path.exists(os.path.join(out, "run_summary.json"))
        time_left = float(dl) - time.time() if dl else 1e9
        if finished and time_left > 15 * 60 and not args.no_eval and not os.path.exists(os.path.join(out, "metrics_best.json")):
            evaluate_run(out, args.root, args.device)
    except Exception:  # keep going; the failure is logged
        traceback.print_exc()
        print(f"[FAILED] {name}")
