"""High-level Python API used by the CLI and the web app.

    from cctempo.api import CCTempoSession
    s = CCTempoSession("work/weinreb/runs/v2_meanfield_s0", root=".")
    s.predict_fate([0, 1, 2], ko_families=["CSF"], exogenous={"G-csf": 1.0})
"""
import os

import numpy as np
import pandas as pd
import torch
import yaml

from .data.dataset import time_index
from .evaluate.fate import fate_bias_from_counts, mc_fate_counts, prepare_field
from .geneops import perturb_genes
from .models import load_run_model
from .runner import classifier_for, load_dataset
from .utils import get_device, load_config


class CCTempoSession:
    def __init__(self, run_dir, root=".", device="auto"):
        self.run_dir, self.root = run_dir, root
        self.cfg = load_config(os.path.join(run_dir, "config_used.yaml"))
        self.device = str(get_device(device))
        self.data = load_dataset(self.cfg, root, self.cfg.get("lr_method"))
        self.model, _ = load_run_model(run_dir, self.data, self.device)
        self.clf = classifier_for(self.cfg, self.data, root)
        ev = self.cfg["eval"]
        self.t0 = float(ev.get("start_time", self.data["times"][0]))
        self.t1 = float(ev.get("end_time", self.data["times"][-1]))
        self.pos, self.neg = ev.get("pos"), ev.get("neg")

    @property
    def families(self):
        return sorted(set(self.model.families)) if self.model.K else []

    @property
    def cytokines(self):
        return list(getattr(self.model, "cytokines", []) or [])

    def start_cells(self, celltype=None):
        i0 = time_index(self.data, self.t0)
        ct = np.asarray(self.data["celltype"][i0])
        return np.where(ct == celltype)[0] if celltype else np.arange(len(ct))

    def predict_fate(self, cells, ko_families=(), exogenous=None, gene_set=None, gene_value=None, n_copies=500, seed=0):
        m, d = self.model, self.data
        m.reset_masks()
        for fam in ko_families or []:
            m.ko_mask[[k for k, f in enumerate(m.families) if f == fam]] = 0.0
        exog = None
        if m.uses_field and exogenous:
            e = m.exogenous_field(exogenous).detach()
            exog = lambda t: e  # noqa: E731
        i0 = time_index(d, self.t0)
        x = d["x"][i0][torch.from_numpy(np.asarray(cells))].to(self.device)
        if gene_set:
            x, _ = perturb_genes(x, d, gene_set, gene_value)
        y = d["v1"][i0][torch.from_numpy(np.asarray(cells))].to(self.device) if m.mode == "v1" else None
        field = prepare_field(m, d, self.t0, self.t1, 0.1, 3000, exog_fn=exog, seed=seed, device=self.device)
        counts = mc_fate_counts(m, x, self.t0, self.t1, 0.1, self.clf, n_copies, field_traj=field, exog_fn=exog,
                                y_prog=y, seed=seed, device=self.device)
        m.reset_masks()
        frac = counts / counts.sum(1, keepdims=True)
        df = pd.DataFrame(frac, columns=self.clf.classes)
        df.insert(0, "cell", np.asarray(cells))
        if self.pos in self.clf.classes and self.neg in self.clf.classes:
            df["fate_bias"] = fate_bias_from_counts(counts, self.clf, self.pos, self.neg)[0]
        return df
