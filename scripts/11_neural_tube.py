"""Step 11: neural tube (GSE236512; the RNA system of Cislo et al., PNAS 2025).

--prepare : download (~1.3 GB), inspect, total = spliced + unspliced, filter, log-normalise, detect
            the time column, annotate (existing labels or marker programmes), save .h5ad.
            Then run 02_liana_ccc.py and 03_build_dataset.py with --config neural_tube.yaml.
--analyze : for a trained run:
   composition.csv     predicted vs observed type fractions per day (incl. p3 / FP)
   endo_hh_ko.csv      fractions when the ENDOGENOUS Hedgehog channel (floor-plate Shh) is knocked out,
                       exogenous SAG kept -> tests the hypothesis that the late p3 -> FP drift the
                       PNAS paper could only fit with a time-changing landscape is driven by Shh
                       secreted by the cells themselves
   landscape_by_day.csv attractor basin masses with the field frozen at each observed day
                       (does the FP basin deepen as the FP-derived field grows?)
   kl_bits.csv         path-integral KL (bits) of simulated vs observed distributions per day

    python scripts/11_neural_tube.py --prepare --root /kaggle/working
    python scripts/11_neural_tube.py --analyze --run v2_meanfield_s0 --root /kaggle/working
"""
import _common
import os
import re

import numpy as np
import pandas as pd

from cctempo.data import neural_tube as nt
from cctempo.runner import P
from cctempo.utils import load_config, save_json

p = _common.parser(__doc__, "neural_tube.yaml")
p.add_argument("--prepare", action="store_true")
p.add_argument("--analyze", action="store_true")
p.add_argument("--run", default="v2_meanfield_s0")
args = p.parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
pc = cfg["prepare"]


def parse_time(v, hours_after_day=pc.get("hours_after_day", 3.0)):
    """Time in days. 'D5' -> 5; '05h' -> hours after Day 3 (replicate r4: 3.21 ... 3.83)."""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    m = re.fullmatch(r"\s*(\d+(\.\d+)?)\s*h\s*", str(v))
    if m:
        return hours_after_day + float(m.group(1)) / 24.0
    m = re.search(r"(\d+(\.\d+)?)", str(v))
    return float(m.group(1)) if m else np.nan


if args.prepare:
    import anndata as ad
    import scanpy as sc
    from cctempo.data.preprocess import normalize_log
    h5 = nt.download(P(cfg, "raw", args.root))
    adata = ad.read_h5ad(h5)
    os.makedirs(P(cfg, "work", args.root), exist_ok=True)
    report = nt.inspect(adata)
    print(report)
    with open(os.path.join(P(cfg, "work", args.root), "inspect.txt"), "w") as f:
        f.write(report)
    nt.total_counts_layer(adata)
    adata = ad.AnnData(X=adata.X, obs=adata.obs.copy(), var=adata.var.copy())  # drop layers (one is keyed None)
    sc.pp.filter_cells(adata, min_counts=pc.get("min_counts", 1000))
    sc.pp.filter_genes(adata, min_cells=10)
    tk = pc["time_key"] if pc["time_key"] != "auto" else nt.find_key(adata, nt.TIME_KEYS)
    if tk is None:
        raise SystemExit("no time column found: look at inspect.txt and set prepare.time_key")
    adata.obs["cct_time"] = adata.obs[tk].map(parse_time).astype(float)
    adata = adata[adata.obs["cct_time"].notna()].copy()
    # keep whole-day samples (replicate 4 has 3.2-3.8 day sub-samples: kept, they are real times)
    normalize_log(adata)
    ck = pc["celltype_key"] if pc["celltype_key"] != "auto" else nt.find_key(adata, nt.TYPE_KEYS)
    if ck is None:
        ck = nt.annotate_by_markers(adata)
    adata.uns["cct_time_key"], adata.uns["cct_celltype_key"] = "cct_time", ck
    print(adata.obs.groupby("cct_time")[ck].value_counts().unstack(fill_value=0).to_string())
    adata.write_h5ad(P(cfg, "h5ad", args.root), compression="gzip")
    save_json({"time_key_source": tk, "celltype_key": ck, "n_cells": adata.n_obs,
               "times": adata.obs["cct_time"].value_counts().sort_index().to_dict()},
              os.path.join(P(cfg, "work", args.root), "prepare_summary.json"))

if args.analyze:
    import torch
    from cctempo.data.dataset import time_index
    from cctempo.evaluate.fate import prepare_field
    from cctempo.evaluate.metrics import observed_composition
    from cctempo.evaluate.pathint import attractors, build_operator, entropy_bits, kl_bits, stationary
    from cctempo.interpret.landscape import basin_table
    from cctempo.models import load_run_model
    from cctempo.runner import classifier_for, load_dataset
    from cctempo.simulate import simulate
    from cctempo.utils import get_device
    dev = str(get_device(args.device))
    data = load_dataset(cfg, args.root)
    model, _ = load_run_model(os.path.join(P(cfg, "runs", args.root), args.run), data, dev)
    clf = classifier_for(cfg, data, args.root)
    out = os.path.join(P(cfg, "results", args.root), f"analysis_{args.run}")
    os.makedirs(out, exist_ok=True)
    times = data["times"]
    t0 = times[0]
    sag = model.exogenous_field(["SAG"]).detach() if model.uses_field else None
    exog = (lambda t: sag) if sag is not None else None
    i0 = time_index(data, t0)
    rng = np.random.default_rng(0)
    idx = torch.from_numpy(rng.choice(data["x"][i0].shape[0], min(4000, data["x"][i0].shape[0]), replace=False))

    def trajectory(label, ko=None):
        model.reset_masks()
        if ko is not None:
            model.ko_mask[ko] = 0.0
        x, g = data["x"][i0][idx].to(dev), data["growth"][i0][idx].to(dev)
        gen = torch.Generator(device=dev).manual_seed(0)
        rows, t_prev = [], t0
        for t in times[1:]:
            o = simulate(model, x, t_prev, t, 0.1, w0=torch.exp(g * (t_prev - t0)), growth=g, exog_fn=exog, generator=gen)
            x, t_prev = o["x"].detach(), t
            lab = np.array(clf.classes)[clf.predict_idx(x).cpu().numpy()]
            for ct, fr in observed_composition(lab).items():
                rows.append({"setting": label, "time": t, "celltype": ct, "fraction": fr})
        model.reset_masks()
        return rows

    rows = trajectory("model")
    for t in times[1:]:
        for ct, fr in observed_composition(data["celltype"][time_index(data, t)]).items():
            rows.append({"setting": "observed", "time": t, "celltype": ct, "fraction": fr})
    pd.DataFrame(rows).to_csv(os.path.join(out, "composition.csv"), index=False)
    if model.K:
        hh = [k for k, (f, e) in enumerate(zip(model.families, data["lr"]["exogenous"])) if f == "HH" and e is None]
        print("endogenous Hedgehog pairs:", [model.pair_names[k] for k in hh])
        if hh:
            pd.DataFrame(trajectory("model", None) + trajectory("endogenous_HH_KO", hh)).to_csv(
                os.path.join(out, "endo_hh_ko.csv"), index=False)
    pi = cfg["eval"]["pathint"]
    op, field = build_operator(model, data, t0, times[-1], clf, pi, exog_fn=exog, device=dev, n_bg=4000)
    lrows, krows = [], []
    for t in times:
        f = None
        if field is not None:
            k = min(int(round((t - t0) / 0.1)), field.shape[0] - 1)
            f = field[k] + (sag if sag is not None else 0)
        pi_ = stationary(op.transition(f))
        roots, labels, mass = attractors(op.P, pi_)
        bt = basin_table(op, pi_, labels, roots)
        bt.insert(0, "field_frozen_at_day", t)
        lrows.append(bt)
    pd.concat(lrows).to_csv(os.path.join(out, "landscape_by_day.csv"), index=False)
    p0 = op.histogram(data["x"][i0].numpy())
    for t in times[1:]:
        Ts = op.transitions(t0, t, 0.1, field_traj=field, exog_fn=exog)
        pobs = op.histogram(data["x"][time_index(data, t)].numpy())
        krows.append({"time": t, "kl_bits": kl_bits(pobs, op.evolve(Ts, p0)[-1]), "entropy_bits": entropy_bits(pobs)})
    pd.DataFrame(krows).to_csv(os.path.join(out, "kl_bits.csv"), index=False)
    print(pd.DataFrame(krows).to_string())
    print("done ->", out)
