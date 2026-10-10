"""CC-Tempo models.

One class covers the whole ablation ladder (``mode``):

* ``prescient``  - U(x) only (re-implementation of PRESCIENT, Yeo et al. 2021).
* ``v1``         - faithful re-implementation of the original CC-Tempo:
                   U = Phi(x) + Psi(y), Psi = Linear-Linear on static communication features y,
                   y evolves by its own (constant) drift. NOTE: dU/dx does not depend on y, so the
                   communication branch cannot change gene-expression trajectories or fates.
* ``local``      - v2, cell-intrinsic coupling: U = Phi(x) + U_c(x; [l(x), r(x)]) where l, r are
                   the cell's own ligand / receptor expression decoded from x.
* ``meanfield``  - v2, population coupling (McKean-Vlasov): U = Phi(x) + U_c(x; s(x, mu_t)) with
                   s_k = gate_k * (alpha * L_k(mu_t) + e_k(t)) * r_k(x),
                   L_k(mu_t) = growth-weighted mean ligand k over the simulated population,
                   e_k(t) = exogenous ligand (e.g. cytokines in the medium).

The drift is v = -grad_x U and cells move by Euler-Maruyama:
    x_{t+dt} = x_t + v dt + sigma * sqrt(dt) * z,   z ~ N(0, I).
"""
import numpy as np
import torch
from torch import nn

from ..ccc.lr_encoder import LREncoder
from .potentials import FiLMPotential, PotentialMLP

MODES = ("prescient", "v1", "local", "meanfield")


def _derangement(n, rng):
    if n < 2:
        return np.arange(n)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def apply_lr_control(lr, genes, context, seed=0):
    """Return (lig_genes, rec_genes) after applying a negative-control transform.

    context: 'lr' (real pairs), 'shuffled' (receptors re-paired at random: same genes, broken
    biology), 'random' (every subunit replaced by a random non-LR gene of the dataset).
    """
    lig = [list(g) for g in lr["lig_genes"]]
    rec = [list(g) for g in lr["rec_genes"]]
    rng = np.random.default_rng(seed)
    if context in ("lr", "off"):
        return lig, rec
    if context == "shuffled":
        perm = _derangement(len(rec), rng)
        return lig, [rec[i] for i in perm]
    if context == "random":
        used = {g for grp in lig + rec for g in grp}
        pool = np.array([g for g in genes if g not in used])
        lig = [list(rng.choice(pool, size=len(g), replace=False)) for g in lig]
        rec = [list(rng.choice(pool, size=len(g), replace=False)) for g in rec]
        return lig, rec
    raise ValueError(f"unknown context control {context}")


class CCTempo(nn.Module):
    def __init__(self, mcfg: dict, x_dim: int, data: dict = None):
        super().__init__()
        self.mode = mcfg.get("mode", "meanfield")
        assert self.mode in MODES, self.mode
        self.sigma = float(mcfg.get("sigma", 0.5))
        self.context = mcfg.get("context", "lr")
        if self.context is False:  # YAML reads a bare `off` as boolean False
            self.context = "off"
        self.grad_through_receptor = bool(mcfg.get("grad_through_receptor", True))
        # centre the field on a reference (the observed start population): channels then carry only
        # *changes* in signalling, so they cannot act as hidden state features, and a knockout
        # (gate = 0) means "clamp this channel at its reference level" (an in-distribution input)
        self.center_field = bool(mcfg.get("center_field", True))
        # centred scale >= scale_floor * reference field, and mean-field signals are softly bounded at
        # +-sig_clip: channels whose observed field barely changes otherwise get scales ~1e-3, and small
        # drifts of the simulated population blow up into signals of ~100 (Weinreb tier-1 diverged)
        self.scale_floor = float(mcfg.get("scale_floor", 0.25))
        self.sig_clip = float(mcfg.get("sig_clip", 3.0))
        self.x_dim = x_dim
        hidden = int(mcfg.get("hidden", 500))
        self.phi = PotentialMLP(x_dim, hidden, int(mcfg.get("layers", 1)), mcfg.get("activation", "softplus"))

        self.v1_dim = 0
        if self.mode == "v1":
            self.v1_dim = int(data["v1"][0].shape[1])
            # faithful to the original code: two Linear layers, no activation
            self.cc = nn.Sequential(nn.Linear(self.v1_dim, hidden), nn.Linear(hidden, 1, bias=False))

        self.K = 0
        if self.mode in ("local", "meanfield"):
            lr = data["lr"]
            lig, rec = apply_lr_control(lr, data["genes"], self.context, int(mcfg.get("context_seed", 0)))
            self.lr = LREncoder(data["pca"]["components"], data["pca"]["mean"], data["scaler"]["mean"],
                                data["scaler"]["scale"], data["genes"], lig, rec,
                                softmin_tau=float(mcfg.get("softmin_tau", 0.1)))
            self.K = self.lr.K
            prior = np.asarray(lr.get("prior", np.ones(self.K)), dtype=np.float32)
            if mcfg.get("gate_init", "prior") == "prior" and self.context == "lr":
                p = np.clip(0.05 + 0.9 * prior, 1e-3, 1 - 1e-3)
            else:
                p = np.full(self.K, 0.5, dtype=np.float32)
            self.gate_logit = nn.Parameter(torch.from_numpy(np.log(p / (1 - p)).astype(np.float32)))
            s_dim = 2 * self.K if self.mode == "local" else self.K
            self.cpot = FiLMPotential(x_dim, s_dim, int(mcfg.get("c_hidden", 256)), mcfg.get("activation", "softplus"))
            self.register_buffer("s_scale", torch.ones(s_dim))
            self.register_buffer("f_ref", torch.zeros(self.K))
            self.register_buffer("ko_mask", torch.ones(self.K))   # ligand-receptor knockouts (receiver side)
            self.register_buffer("rec_mask", torch.ones(self.K))  # receptor knockouts
            exo = lr.get("exogenous", [None] * self.K)
            self.register_buffer("exo_flag", torch.tensor([e is not None for e in exo], dtype=torch.bool))
            # one learnable log-dose per exogenous ligand (cytokine) -- a few PNAS-style knobs
            self.cytokines = sorted({e for e in exo if e is not None})
            self.log_dose = nn.Parameter(torch.zeros(len(self.cytokines))) if self.cytokines else None
            self._exo_index = {c: [k for k, e in enumerate(exo) if e == c] for c in self.cytokines}
            self.families = list(lr.get("family", [f"LR{k}" for k in range(self.K)]))
            self.pair_names = [f"{a}->{b}" for a, b in zip(lr.get("ligand", range(self.K)), lr.get("receptor", range(self.K)))]

    # ------------------------------------------------------------------ communication
    @property
    def uses_field(self):
        return self.mode == "meanfield"

    def gates(self):
        return torch.sigmoid(self.gate_logit) * self.ko_mask

    def ligand_field(self, x, w=None):
        """Growth-weighted mean ligand expression over a population, shape (K,)."""
        l = self.lr.ligand(x)
        if w is None:
            return l.mean(0)
        w = w / w.sum()
        return (w[:, None] * l).sum(0)

    def exogenous_field(self, cytokines, device=None):
        """Field contribution of exogenous ligands present in the medium, shape (K,).

        cytokines: list of names (relative dose 1) or dict {name: relative dose}. The absolute
        scale of each cytokine is a learned parameter exp(log_dose)."""
        e = torch.zeros(self.K, device=device or self.gate_logit.device)
        if not cytokines or self.log_dose is None:
            return e
        items = cytokines.items() if isinstance(cytokines, dict) else [(c, 1.0) for c in cytokines]
        for c, rel in items:
            if c not in self._exo_index:
                continue
            j = self.cytokines.index(c)
            for k in self._exo_index[c]:
                e[k] = e[k] + float(rel) * torch.exp(self.log_dose[j])
        return e

    def signal(self, x, field=None):
        if self.context == "off":
            dim = 2 * self.K if self.mode == "local" else self.K
            return torch.zeros(x.shape[0], dim, device=x.device)
        xr = x if self.grad_through_receptor else x.detach()
        g = self.gates()
        if self.mode == "local":
            l, r = self.lr(xr)
            s = torch.cat([g * l, g * r * self.rec_mask], dim=1)
        else:
            r = self.lr.receptor(xr) * self.rec_mask
            f = field - self.f_ref if self.center_field else field
            s = g * f[None, :] * r / self.s_scale
            if self.sig_clip > 0:  # identity for |s| << sig_clip, saturates beyond
                s = self.sig_clip * torch.tanh(s / self.sig_clip)
            return s
        return s / self.s_scale

    @torch.no_grad()
    def calibrate(self, x, w=None, x_by_time=None):
        """Set the reference field and per-channel scales so that typical received signals are O(1).

        x: pooled cells (any times); x_by_time: list of per-time cell tensors, first = start time."""
        if self.mode not in ("local", "meanfield"):
            return
        l, r = self.lr(x)
        if self.mode == "local":
            sc = torch.cat([l.mean(0), r.mean(0)])
        else:
            rm = r.mean(0)
            exo = rm  # reference exogenous level = 1 log-expression unit
            if self.center_field and x_by_time:
                fields = torch.stack([self.ligand_field(xt) for xt in x_by_time])
                self.f_ref.copy_(fields[0])
                dev = (fields - fields[0]).abs().max(0).values
                endo = (torch.maximum(dev, self.scale_floor * fields[0]) + 1e-3) * rm
            else:
                endo = self.ligand_field(x, w) * rm
            sc = torch.where(self.exo_flag, torch.maximum(endo, exo), endo)
        self.s_scale.copy_(sc.clamp_min(1e-4))

    # ------------------------------------------------------------------ dynamics
    def potential(self, x, y=None, s=None):
        U = self.phi(x)
        if self.mode == "v1":
            U = U + self.cc(y).squeeze(-1)
        if s is not None:
            U = U + self.cpot(x, s)
        return U

    def drift(self, x, y=None, field=None, create_graph=False):
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            inputs = [x]
            if self.mode == "v1":
                if y is None:  # v1: dU/dx does not depend on y, so zeros are exact for the x-drift
                    y = torch.zeros(x.shape[0], self.v1_dim, device=x.device)
                if not y.requires_grad:
                    y = y.detach().requires_grad_(True)
                inputs.append(y)
            s = None
            if self.mode == "local":
                s = self.signal(x)
            elif self.mode == "meanfield":
                # the field reaches here without dependence on particle positions (see simulate), so it
                # keeps its gradient w.r.t. parameters (exogenous doses) but creates no 'sender force'
                s = self.signal(x, field)
            U = self.potential(x, y, s)
            grads = torch.autograd.grad(U.sum(), inputs, create_graph=create_graph)
        vx = -grads[0]
        vy = -grads[1] if self.mode == "v1" else None
        if not create_graph:
            vx = vx.detach()
            vy = vy.detach() if vy is not None else None
        return vx, vy

    def step(self, x, y, field, dt, z, create_graph=False):
        vx, vy = self.drift(x, y, field, create_graph)
        x_new = x + vx * dt + self.sigma * (dt ** 0.5) * z
        y_new = y + vy * dt if (vy is not None and y is not None) else y
        return x_new, y_new

    # ------------------------------------------------------------------ utilities
    def reset_masks(self):
        if self.K:
            self.ko_mask.fill_(1.0)
            self.rec_mask.fill_(1.0)

    def describe(self):
        d = {"mode": self.mode, "context": self.context, "K": self.K,
             "params": sum(p.numel() for p in self.parameters())}
        if self.K:
            d["gates"] = dict(zip(self.pair_names, torch.sigmoid(self.gate_logit).detach().cpu().numpy().round(4).tolist()))
            if self.log_dose is not None:
                d["doses"] = dict(zip(self.cytokines, torch.exp(self.log_dose).detach().cpu().numpy().round(4).tolist()))
        return d


def build_model(mcfg: dict, data: dict, device="cpu", calibrate=True):
    x_dim = int(data["x"][0].shape[1])
    model = CCTempo(mcfg, x_dim, data).to(device)
    if calibrate and model.mode in ("local", "meanfield"):
        gen = torch.Generator().manual_seed(0)

        def _sub(t, n):
            t = t.float()
            return t[torch.randperm(t.shape[0], generator=gen)[:n]] if t.shape[0] > n else t

        xs = _sub(torch.cat([t.float() for t in data["x"]], 0), 20000)
        per_time = [_sub(t, 8000).to(device) for t in data["x"]]
        model.calibrate(xs.to(device), x_by_time=per_time)
    return model


def load_run_model(run_dir, data, device="cpu", which="best"):
    import os
    ck = torch.load(os.path.join(run_dir, f"model_{which}.pt"), map_location=device, weights_only=False)
    model = build_model(ck["model_cfg"], data, device=device, calibrate=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
