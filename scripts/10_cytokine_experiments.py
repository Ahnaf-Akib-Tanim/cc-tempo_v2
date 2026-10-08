"""Step 10: cytokine-perturbation experiments (Weinreb et al. 2020, Experiment 3).

--evidence       DATA ONLY (no model, ~3 MB download): clones split across media. For every pair of
                 media, compare the fate bias of the SAME progenitor clones (Wilcoxon signed-rank)
                 and the Day-4/6 composition per medium. Direct lineage-traced evidence that
                 ligand signalling changes fate (Reviewer 1a).
--dose-response  MODEL: sweep the exogenous dose of each cytokine for a trained v2 model and report
                 predicted fate composition (compare the direction with --evidence).
Leave-one-condition-out training is in configs/experiments.yaml (cytokine tier2); its held-out
metrics come out of 07_evaluate.py.

    python scripts/10_cytokine_experiments.py --config cytokine.yaml --evidence --root .
    python scripts/10_cytokine_experiments.py --config cytokine.yaml --dose-response --run v2_meanfield_s0 --root /kaggle/working
"""
import _common
import itertools
import os
import shutil
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cctempo.data.weinreb import BASE, KINDS, clonal_fate_bias, read_mtx
from cctempo.runner import P
from cctempo.utils import load_config

p = _common.parser(__doc__, "cytokine.yaml")
p.add_argument("--evidence", action="store_true")
p.add_argument("--dose-response", action="store_true")
p.add_argument("--run", default="v2_meanfield_s0")
args = p.parse_args()
cfg = load_config(_common.cfg_path(args.config), args.set)
ev, pc = cfg["eval"], cfg["prepare"]
res = P(cfg, "results", args.root)
os.makedirs(res, exist_ok=True)

if args.evidence:
    raw = P(cfg, "raw", args.root)
    os.makedirs(raw, exist_ok=True)
    files = {}
    for part in ("metadata.txt.gz", "clone_matrix.mtx.gz"):
        fn = f"{KINDS['cytokine']}_{part}"
        dst = os.path.join(raw, fn)
        if not os.path.exists(dst):
            with urllib.request.urlopen(BASE + fn) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
        files[part] = dst
    meta = pd.read_csv(files["metadata.txt.gz"], sep="\t")
    C = read_mtx(files["clone_matrix.mtx.gz"])
    tk, ck, cond_key = pc["time_key"], pc["celltype_key"], pc["condition_key"]
    comp = (meta[meta[tk] > 2].groupby([cond_key, tk])[ck].value_counts(normalize=True).rename("fraction").reset_index())
    comp.to_csv(os.path.join(res, "evidence_composition.csv"), index=False)
    per = {}
    for cond in ev["conditions"]:
        prog, bias, npos, nneg = clonal_fate_bias(meta, C, tk, ck, ev["start_time"], ev["pos"], ev["neg"], ev["undiff"], cond_key, cond)
        per[cond] = pd.Series(bias, index=prog)
    rows = []
    for a, b in itertools.combinations(ev["conditions"], 2):
        shared = per[a].index.intersection(per[b].index)
        if len(shared) < 10:
            continue
        x, y = per[a].loc[shared].values, per[b].loc[shared].values
        st = wilcoxon(x, y)
        rows.append({"medium_a": a, "medium_b": b, "n_shared_progenitors": len(shared),
                     "mean_neu_bias_a": x.mean(), "mean_neu_bias_b": y.mean(), "mean_diff_a_minus_b": (x - y).mean(),
                     "wilcoxon_p": st.pvalue})
    tab = pd.DataFrame(rows).sort_values("wilcoxon_p")
    tab.to_csv(os.path.join(res, "evidence_clone_matched.csv"), index=False)
    print(tab.round(4).to_string(index=False))
    wide = comp[comp[tk] == 6].pivot(index=cond_key, columns=ck, values="fraction").fillna(0).round(3)
    print("\nDay-6 composition per medium\n", wide.to_string())

if args.dose_response:
    import torch
    from cctempo.interpret.perturb import FateRunner
    from cctempo.models import load_run_model
    from cctempo.runner import classifier_for, load_dataset
    from cctempo.utils import get_device
    dev = str(get_device(args.device))
    data = load_dataset(cfg, args.root)
    model, _ = load_run_model(os.path.join(P(cfg, "runs", args.root), args.run), data, dev)
    clf = classifier_for(cfg, data, args.root)
    clonal = data["eval"]["clonal"]["full cocktail"]
    prog = np.asarray(clonal["prog_idx"])[:400]
    runner = FateRunner(model, data, clf, prog, ev["start_time"], ev["end_time"], n_copies=100, device=dev,
                        pos=ev["pos"], neg=ev["neg"])
    rows = []
    for cyt in model.cytokines:
        for rel in [0.0, 0.5, 1.0, 2.0, 4.0]:
            e = model.exogenous_field({cyt: rel}).detach()
            summ, _ = runner.summarize(runner.counts(exog_fn=lambda t, e=e: e))
            rows.append({"cytokine": cyt, "relative_dose": rel, **summ})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(res, f"dose_response_{args.run}.csv"), index=False)
    print(df.round(3).to_string(index=False))
