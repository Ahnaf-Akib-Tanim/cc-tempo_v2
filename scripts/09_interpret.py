"""Step 9: interpretation and in-silico experiments on a trained model (default: v2_meanfield seed 0).

Writes CSVs to paths.results/interpret_<run>/:
  ko_family.csv, ko_pairs.csv     LR knockout scan (receiver side), paired effects +- SE
  sender_ablation.csv             silence ligands of one lineage (indirect effect, Reviewer 1b)
  temporal_windows.csv            block top families in D2-4 vs D4-6 (window of opportunity)
  tf_perturbation.csv             true lineage TFs up/down (+ the v1 marker sets for comparison)
  genes_fate_attr_day2.csv        progenitor genes driving fate (IG through the SDE)
  genes_potential_attr_day6.csv   original-paper style attribution on the landscape value
  gene_de_validation.csv          do top genes differ between lineage-traced mono- vs neu-biased clones?
  signal_to_tf.csv                TF programmes changed by the top ligand family (LR -> TF -> fate)
  bifurcation.csv                 attractors / basin masses vs communication strength alpha

    python scripts/09_interpret.py --config weinreb.yaml --root /kaggle/working --run v2_meanfield_s0
"""
import _common
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import ranksums

from cctempo.data.dataset import time_index
from cctempo.data.genesets import TFS, V1_LINEAGE_GENES
from cctempo.evaluate.fate import prepare_field
from cctempo.evaluate.pathint import build_operator
from cctempo.geneops import decode_scaled
from cctempo.interpret.attribution import fate_attribution, potential_attribution, rank_genes
from cctempo.interpret.landscape import communication_bifurcation
from cctempo.interpret.perturb import (FateRunner, gene_perturbation, lr_knockout_scan, sender_ablation,
                                       temporal_block_scan)
from cctempo.models import load_run_model
from cctempo.runner import P, classifier_for, load_dataset
from cctempo.simulate import simulate
from cctempo.utils import get_device, load_config, save_json

p = _common.parser(__doc__, "weinreb.yaml")
p.add_argument("--run", default="v2_meanfield_s0")
p.add_argument("--n-prog", type=int, default=600, help="progenitors used (subsample for speed)")
p.add_argument("--skip", nargs="*", default=[], help="steps to skip: ko sender windows tf attr signal bif")
args = p.parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
dev = str(get_device(args.device))
run_dir = os.path.join(P(cfg, "runs", args.root), args.run)
import yaml  # noqa: E402
with open(os.path.join(run_dir, "config_used.yaml")) as f:
    used = yaml.safe_load(f)
data = load_dataset(cfg, args.root, used.get("lr_method"))
model, _ = load_run_model(run_dir, data, dev)
clf = classifier_for(cfg, data, args.root)
ev = cfg["eval"]
t0, t1 = float(ev["start_time"]), float(ev["end_time"])
pos, neg = ev["pos"], ev["neg"]
clonal = data["eval"]["clonal"]
if "prog_idx" not in clonal:  # conditional dataset -> use the first condition
    cond0 = next(iter(clonal))
    clonal = clonal[cond0]
rng = np.random.default_rng(0)
sel = rng.choice(len(clonal["prog_idx"]), min(args.n_prog, len(clonal["prog_idx"])), replace=False)
prog, bias_true = np.asarray(clonal["prog_idx"])[sel], np.asarray(clonal["bias"])[sel]
out = os.path.join(P(cfg, "results", args.root), f"interpret_{args.run}")
os.makedirs(out, exist_ok=True)
runner = FateRunner(model, data, clf, prog, t0, t1, n_copies=int(ev.get("ko_copies", 200)),
                    n_bg=int(ev.get("n_bg", 5000)), device=dev, pos=pos, neg=neg)
base_counts = runner.counts()
summary, _ = runner.summarize(base_counts)
save_json({"unperturbed": summary, "model": model.describe()}, os.path.join(out, "unperturbed.json"))
print("unperturbed:", summary)

if model.K and "ko" not in args.skip:
    ko_f = lr_knockout_scan(runner, "family"); ko_f.to_csv(os.path.join(out, "ko_family.csv"), index=False)
    print(ko_f.head(10).to_string())
    ko_p = lr_knockout_scan(runner, "pair"); ko_p.to_csv(os.path.join(out, "ko_pairs.csv"), index=False)
    top_fams = ko_f["group"].head(3).tolist()
else:
    top_fams = []

if model.uses_field and "sender" not in args.skip:
    rows = []
    for ct in [c for c in clf.classes if c != ev.get("undiff")]:
        rows.append({"silenced_sender": ct, **sender_ablation(runner, [ct])})
    pd.DataFrame(rows).to_csv(os.path.join(out, "sender_ablation.csv"), index=False)

if model.uses_field and top_fams and "windows" not in args.skip:
    mid = data["times"][len(data["times"]) // 2]
    tabs = []
    for fam in top_fams:
        ks = [k for k, f in enumerate(model.families) if f == fam]
        tb = temporal_block_scan(runner, ks, [(t0, mid), (mid, t1), (t0, t1)])
        tb.insert(0, "family", fam)
        tabs.append(tb)
    pd.concat(tabs).to_csv(os.path.join(out, "temporal_windows.csv"), index=False)

if "tf" not in args.skip:
    tabs = []
    for name, sets in (("TF", TFS), ("v1_markers", V1_LINEAGE_GENES)):
        for lineage, genes in sets.items():
            tb = gene_perturbation(runner, genes, [-2.5, -1.0, 1.0, 2.5])
            tb.insert(0, "set", f"{name}:{lineage}")
            tabs.append(tb)
    pd.concat(tabs).to_csv(os.path.join(out, "tf_perturbation.csv"), index=False)

if "attr" not in args.skip:
    i0 = time_index(data, t0)
    field = prepare_field(model, data, t0, t1, 0.1, int(ev.get("n_bg", 5000)), device=dev)
    sub = prog[:128]
    x0 = data["x"][i0][torch.from_numpy(sub)].to(dev)
    A = fate_attribution(model, data, clf, x0, t0, t1, 0.1, pos, neg, field_traj=field)
    pd.DataFrame(rank_genes(A, data["genes"], 40), columns=["gene", "mean_abs_attr", "sign_to_" + pos]).to_csv(
        os.path.join(out, "genes_fate_attr_day2.csv"), index=False)
    g = torch.Generator(device=dev).manual_seed(0)
    xe = simulate(model, x0.repeat_interleave(4, 0), t0, t1, 0.1, field_traj=field, generator=g)["x"].detach()
    lab = np.array(clf.classes)[clf.predict_idx(xe).cpu().numpy()]
    rows = []
    for ct in (pos, neg):
        m = lab == ct
        if m.sum() < 5:
            continue
        Ap = potential_attribution(model, data, xe[torch.from_numpy(m)], field[-1] if field is not None else None)
        for gname, sc_, sg in rank_genes(Ap, data["genes"], 25):
            rows.append({"lineage": ct, "gene": gname, "mean_abs_attr": sc_, "sign": sg})
    pd.DataFrame(rows).to_csv(os.path.join(out, "genes_potential_attr_day6.csv"), index=False)
    # validation with lineage-tracing labels (not with predicted labels)
    G = decode_scaled(data["x"][i0][torch.from_numpy(prog)], data).numpy()
    top = pd.read_csv(os.path.join(out, "genes_fate_attr_day2.csv"))["gene"].tolist()
    mono_b, neu_b = bias_true < 0.4, bias_true > 0.6
    de = []
    for gname in top:
        j = data["genes"].index(gname)
        st = ranksums(G[neu_b, j], G[mono_b, j])
        de.append({"gene": gname, "stat_neu_vs_mono_clones": st.statistic, "p": st.pvalue})
    pd.DataFrame(de).to_csv(os.path.join(out, "gene_de_validation.csv"), index=False)

if model.uses_field and top_fams and "signal" not in args.skip:
    try:
        from cctempo.interpret.tf_activity import load_collectri, tf_matrix
        tfs, W = tf_matrix(load_collectri("mouse"), data["genes"])
        f_on = prepare_field(model, data, t0, t1, 0.1, 3000, device=dev)
        ks = [k for k, f in enumerate(model.families) if f == top_fams[0]]
        model.ko_mask[ks] = 0.0
        f_off = prepare_field(model, data, t0, t1, 0.1, 3000, device=dev)
        x0 = data["x"][time_index(data, t0)][torch.from_numpy(prog[:200])].to(dev)
        model.reset_masks()
        model.ko_mask[ks] = 0.0
        g = torch.Generator(device=dev).manual_seed(0)
        xr = x0.repeat_interleave(30, 0)
        x_off = simulate(model, xr, t0, t1, 0.1, field_traj=f_off, generator=g)["x"]
        model.reset_masks()
        g = torch.Generator(device=dev).manual_seed(0)
        x_on = simulate(model, xr, t0, t1, 0.1, field_traj=f_on, generator=g)["x"]
        d = ((decode_scaled(x_on, data) - decode_scaled(x_off, data)) @ W.to(dev).T).view(len(x0), 30, -1).mean(1)
        se = d.std(0) / np.sqrt(d.shape[0])
        pd.DataFrame({"tf": tfs, "delta_activity_on_minus_ko": d.mean(0).cpu().numpy(),
                      "t": (d.mean(0) / se.clamp_min(1e-8)).cpu().numpy(), "family": top_fams[0]}).sort_values(
            "t", key=np.abs, ascending=False).to_csv(os.path.join(out, "signal_to_tf.csv"), index=False)
    except Exception as e:  # noqa: BLE001
        print("[signal->TF skipped]", e)

if "bif" not in args.skip:
    pi = cfg["eval"]["pathint"]
    op, field = build_operator(model, data, t0, t1, clf, pi, device=dev, n_bg=int(ev.get("n_bg", 5000)))
    fend = field[-1] if field is not None else None
    bif, details = communication_bifurcation(model, op, fend, [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    bif.to_csv(os.path.join(out, "bifurcation.csv"), index=False)
    for a, d in details.items():
        d["basins"].to_csv(os.path.join(out, f"basins_alpha{a}.csv"), index=False)
    np.savez_compressed(os.path.join(out, "landscape_alpha1.npz"), P=op.P[:, :2], U_eff=details[1.0]["U_eff"],
                        celltype=op.celltype)
    print(bif.to_string())
print("done ->", out)
