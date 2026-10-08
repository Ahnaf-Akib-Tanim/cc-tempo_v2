"""Step 8: external baselines on the same data, evaluated with the same metrics.

  wot   : Waddington-OT style unbalanced OT fate + displacement interpolation (POT)
  cfm   : OT conditional flow matching (neural ODE), deterministic + small-noise SDE variant
  prescient_official (optional): see cctempo/baselines/prescient_official.py

    python scripts/08_baselines.py --config weinreb.yaml --root /kaggle/working --methods wot cfm
"""
import _common
import os

import numpy as np
import pandas as pd
import torch

from cctempo.baselines.cfm import integrate, train_cfm
from cctempo.baselines.ot_baselines import wot_fate, wot_interpolate
from cctempo.data.dataset import cells, time_index
from cctempo.evaluate.metrics import distribution_metrics, fate_metrics
from cctempo.runner import P, _eval_sets, classifier_for, load_dataset
from cctempo.utils import get_device, load_config

p = _common.parser(__doc__, "weinreb.yaml")
p.add_argument("--methods", nargs="*", default=["wot", "cfm"])
p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
p.add_argument("--cfm-iters", type=int, default=20000)
args = p.parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
dev = str(get_device(args.device))
data = load_dataset(cfg, args.root)
clf = classifier_for(cfg, data, args.root)
ev = cfg["eval"]
t0 = float(ev.get("start_time", data["times"][0]))
t1 = float(ev.get("end_time", data["times"][-1]))
pos, neg = ev["pos"], ev["neg"]
out_dir = P(cfg, "results", args.root)
os.makedirs(out_dir, exist_ok=True)
rows = []
times = [t for t in data["times"] if t >= t0]
for seed in args.seeds:
    for cond, prog, target in _eval_sets(data, cfg):
        prog = np.asarray(prog)
        if "wot" in args.methods:
            fb = wot_fate(data, clf, prog, pos, neg, seed=seed, times=times, condition=cond)
            rows.append({"model": "WOT", "seed": seed, "condition": str(cond), **{f"mc_{k}": v for k, v in fate_metrics(fb, target).items()}})
            np.save(os.path.join(out_dir, f"preds_WOT_{cond}_s{seed}.npy"), fb)
        if "cfm" in args.methods:
            Xs = []
            for k, t in enumerate(times):
                m = cells(data, t, "train", cond if k > 0 else None)
                Xs.append(data["x"][time_index(data, t)][torch.from_numpy(np.where(m)[0])].numpy())
            model = train_cfm(Xs, times, n_iter=args.cfm_iters, device=dev, seed=seed)
            x0 = data["x"][time_index(data, t0)][torch.from_numpy(prog)].to(dev)
            for sde in (0.0, 0.5):
                reps = 1 if sde == 0 else 50
                xe = integrate(model, x0.repeat_interleave(reps, 0), t0, t1, sde_sigma=sde, seed=seed)
                pr = clf.predict_proba(xe.cpu().numpy()).reshape(len(prog), reps, -1).mean(1)
                fb = pr[:, clf.index(pos)] / np.clip(pr[:, clf.index(pos)] + pr[:, clf.index(neg)], 1e-12, None)
                name = "OT-CFM" if sde == 0 else "OT-CFM-SDE"
                rows.append({"model": name, "seed": seed, "condition": str(cond), **{f"mc_{k}": v for k, v in fate_metrics(fb, target).items()}})

# held-out time interpolation (only for datasets with >= 3 time points)
if len(times) >= 3:
    mid = times[1]
    m0 = cells(data, times[0], "all"); m2 = cells(data, times[2], "all")
    X0 = data["x"][time_index(data, times[0])][torch.from_numpy(np.where(m0)[0])].numpy()
    X2 = data["x"][time_index(data, times[2])][torch.from_numpy(np.where(m2)[0])].numpy()
    Xm = data["x"][time_index(data, mid)].numpy()
    rng = np.random.default_rng(0)
    i0 = rng.choice(len(X0), min(6000, len(X0)), replace=False)
    i2 = rng.choice(len(X2), min(6000, len(X2)), replace=False)
    g0 = data["growth"][time_index(data, times[0])].numpy()[np.where(m0)[0]][i0]
    s = (mid - times[0]) / (times[2] - times[0])
    if "wot" in args.methods:
        Xi = wot_interpolate(X0[i0], X2[i2], np.exp(g0 * (times[2] - times[0])), s)
        rows.append({"model": "WOT", "seed": 0, "condition": f"interp_t{mid}", **{f"{k}_t{mid}": v for k, v in distribution_metrics(Xi, Xm).items()}})
    if "cfm" in args.methods:
        model = train_cfm([X0[i0], X2[i2]], [times[0], times[2]], n_iter=args.cfm_iters, device=dev, seed=0)
        Xi = integrate(model, torch.as_tensor(X0[i0], device=dev), times[0], mid).cpu().numpy()
        rows.append({"model": "OT-CFM", "seed": 0, "condition": f"interp_t{mid}", **{f"{k}_t{mid}": v for k, v in distribution_metrics(Xi, Xm).items()}})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out_dir, "baselines_long.csv"), index=False)
print(df.groupby(["model", "condition"]).mean(numeric_only=True).round(4).to_string())
