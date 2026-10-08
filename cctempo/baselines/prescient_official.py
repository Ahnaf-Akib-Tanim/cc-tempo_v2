"""Optional: run the *official* PRESCIENT package on our data and evaluate it with our code.

Our `mode: prescient` re-implements PRESCIENT's AutoGenerator and training loop with the same
defaults. Reviewers may still ask for the official package, so:
  1. export_prescient(data, path)        -> PRESCIENT-format data.pt
  2. `prescient train_model -i path --out_dir DIR --weight_name kegg-growth --seed S`
  3. OfficialPrescient.load(DIR)         -> wrapper usable by mc_fate_counts / pathint
UNTESTED against every PRESCIENT version: if its internals changed, adapt `load`.
"""
import glob
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


def export_prescient(data, path):
    xp = [x.numpy() for x in data["x"]]
    torch.save({"xp": [torch.from_numpy(x) for x in xp], "x": [torch.from_numpy(x) for x in xp],
                "xu": [torch.zeros(x.shape[0], 2) for x in xp], "y": [float(t) for t in data["times"]],
                "w": [g.numpy() for g in data["growth"]], "celltype": np.concatenate(data["celltype"]),
                "genes": data["genes"], "tps": np.concatenate([[t] * len(x) for t, x in zip(data["times"], xp)])}, path)


class OfficialPrescient(nn.Module):
    """Adapter exposing the CCTempo interface (step / drift / uses_field / sigma)."""

    uses_field = False
    mode = "prescient"
    K = 0

    def __init__(self, net, sigma=0.5):
        super().__init__()
        self.net, self.sigma = net, sigma

    @classmethod
    def load(cls, run_dir, device="cpu", sigma=0.5):
        from prescient.train.model import AutoGenerator
        cfg = SimpleNamespace(**torch.load(os.path.join(run_dir, "config.pt"), weights_only=False))
        net = AutoGenerator(cfg)
        ck = sorted(glob.glob(os.path.join(run_dir, "train.*.pt")))[-1]
        net.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model_state_dict"])
        return cls(net.to(device), sigma)

    def drift(self, x, y=None, field=None, create_graph=False):
        with torch.enable_grad():
            v = self.net._drift(x.detach().requires_grad_(True))
        return v.detach(), None

    def step(self, x, y, field, dt, z, create_graph=False):
        return self.net._step(x, dt=dt, z=z * self.sigma).detach(), y
