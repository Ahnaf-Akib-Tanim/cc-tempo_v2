"""In-silico experiments.

All experiments reuse the same random numbers as the unperturbed run (common random numbers),
so the reported effect is a paired difference with a small standard error even with few copies.

* gene / TF perturbation in progenitors (PRESCIENT-style z-score setting)
* ligand-receptor knockout scan (receiver side; one pair or one ligand family at a time)
* receptor knockout
* sender ablation: silence the ligands produced by one lineage (Reviewer 1(b): indirect effects)
* temporal blocking windows (window of opportunity; order / non-commutativity tests)
* exogenous ligand dose-response (cytokines; morphogen-like control parameter)
"""
import numpy as np
import pandas as pd
import torch

from ..data.dataset import time_index
from ..evaluate.fate import fate_bias_from_counts, mc_fate_counts, prepare_field


class FateRunner:
    """Bundles the fixed pieces of a fate experiment (model, data, classifier, progenitors)."""

    def __init__(self, model, data, clf, prog_idx, t0, t1, dt=0.1, n_copies=200, n_bg=5000, seed=0,
                 device="cpu", pos="Neutrophil", neg="Monocyte", condition=None, exog_fn=None):
        self.model, self.data, self.clf = model, data, clf
        self.i0 = time_index(data, t0)
        self.prog_idx = np.asarray(prog_idx)
        self.x_prog = data["x"][self.i0][torch.from_numpy(self.prog_idx)].to(device)
        self.y_prog = data["v1"][self.i0][torch.from_numpy(self.prog_idx)].to(device) if model.mode == "v1" else None
        self.t0, self.t1, self.dt = t0, t1, dt
        self.n_copies, self.n_bg, self.seed, self.device = n_copies, n_bg, seed, device
        self.pos, self.neg, self.condition, self.exog_fn = pos, neg, condition, exog_fn

    def counts(self, x_prog=None, exog_fn="default", alpha_fn=None, field_weight_fn=None):
        exog_fn = self.exog_fn if exog_fn == "default" else exog_fn
        field = prepare_field(self.model, self.data, self.t0, self.t1, self.dt, self.n_bg, exog_fn, alpha_fn,
                              self.seed, self.device, condition=None, field_weight_fn=field_weight_fn)
        return mc_fate_counts(self.model, self.x_prog if x_prog is None else x_prog, self.t0, self.t1, self.dt,
                              self.clf, self.n_copies, field_traj=field, exog_fn=exog_fn, alpha_fn=alpha_fn,
                              y_prog=self.y_prog, seed=self.seed, device=self.device)

    def summarize(self, counts):
        bias, mask = fate_bias_from_counts(counts, self.clf, self.pos, self.neg)
        frac = counts / counts.sum(1, keepdims=True)
        out = {"fate_bias_mean": float(bias.mean())}
        for j, c in enumerate(self.clf.classes):
            out[f"pct_{c}"] = float(100 * frac[:, j].mean())
        return out, bias

    def paired_effect(self, base_counts, pert_counts):
        b0, _ = fate_bias_from_counts(base_counts, self.clf, self.pos, self.neg)
        b1, _ = fate_bias_from_counts(pert_counts, self.clf, self.pos, self.neg)
        d = b1 - b0
        f0 = base_counts / base_counts.sum(1, keepdims=True)
        f1 = pert_counts / pert_counts.sum(1, keepdims=True)
        out = {"d_fate_bias": float(d.mean()), "se_d_fate_bias": float(d.std(ddof=1) / np.sqrt(len(d)))}
        for j, c in enumerate(self.clf.classes):
            dj = 100 * (f1[:, j] - f0[:, j])
            out[f"d_pct_{c}"] = float(dj.mean())
            out[f"se_pct_{c}"] = float(dj.std(ddof=1) / np.sqrt(len(dj)))
        return out


def lr_groups(model, level="family"):
    if level == "pair":
        return {n: [k] for k, n in enumerate(model.pair_names)}
    groups = {}
    for k, f in enumerate(model.families):
        groups.setdefault(f, []).append(k)
    return groups


def lr_knockout_scan(runner: FateRunner, level="family"):
    """Effect on fate of removing each LR pair / ligand family on the receiver side."""
    model = runner.model
    base = runner.counts()
    rows = []
    for name, ks in lr_groups(model, level).items():
        model.reset_masks()
        model.ko_mask[ks] = 0.0
        eff = runner.paired_effect(base, runner.counts())
        rows.append({"group": name, "n_pairs": len(ks), **eff})
    model.reset_masks()
    df = pd.DataFrame(rows)
    df["abs_effect"] = df["d_fate_bias"].abs() + sum(df[c].abs() for c in df.columns if c.startswith("d_pct_")) / 100
    return df.sort_values("abs_effect", ascending=False).reset_index(drop=True)


def receptor_knockout(runner: FateRunner, pair_idx):
    model = runner.model
    base = runner.counts()
    model.reset_masks()
    model.rec_mask[pair_idx] = 0.0
    eff = runner.paired_effect(base, runner.counts())
    model.reset_masks()
    return eff


def sender_ablation(runner: FateRunner, celltypes):
    """Silence ligand production of cells currently classified as one of `celltypes`."""
    clf = runner.clf
    bad = torch.tensor([clf.index(c) for c in celltypes if c in clf.classes], device=runner.device)

    def weight_fn(x):
        pred = clf.predict_idx(x)
        return (~torch.isin(pred, bad)).float()

    base = runner.counts()
    return runner.paired_effect(base, runner.counts(field_weight_fn=weight_fn))


def window_alpha(model, pair_idx, t_start, t_end):
    """alpha_fn that blocks the given pairs only for t in [t_start, t_end)."""
    def fn(t):
        a = torch.ones(model.K, device=model.gate_logit.device)
        if t_start <= t < t_end:
            a[pair_idx] = 0.0
        return a
    return fn


def temporal_block_scan(runner: FateRunner, pair_idx, windows):
    base = runner.counts()
    rows = []
    for (a, b) in windows:
        eff = runner.paired_effect(base, runner.counts(alpha_fn=window_alpha(runner.model, pair_idx, a, b)))
        rows.append({"window": f"{a}-{b}", **eff})
    return pd.DataFrame(rows)


def exogenous_dose_response(runner: FateRunner, pair_idx, doses):
    model = runner.model
    rows = []
    base = runner.counts(exog_fn=None)
    for d in doses:
        e = torch.zeros(model.K, device=runner.device)
        e[pair_idx] = float(d)
        cnt = runner.counts(exog_fn=lambda t, e=e: e)
        summ, _ = runner.summarize(cnt)
        rows.append({"dose": d, **summ, **{f"vs0_{k}": v for k, v in runner.paired_effect(base, cnt).items()}})
    return pd.DataFrame(rows)


def gene_perturbation(runner: FateRunner, genes, values, mode="set"):
    from ..geneops import perturb_genes
    base = runner.counts()
    rows = []
    for v in values:
        x2, info = perturb_genes(runner.x_prog, runner.data, genes, v, mode)
        cnt = runner.counts(x_prog=x2)
        summ, _ = runner.summarize(cnt)
        rows.append({"value": v, **summ, **runner.paired_effect(base, cnt), "retained": info["retained_fraction"],
                     "missing": ",".join(info["missing"])})
    return pd.DataFrame(rows)
