"""Path-integral operator on the data point cloud (after Cislo, Deles, Briscoe & Siggia, PNAS 2025).

Instead of Monte-Carlo simulating thousands of copies per progenitor, we discretise the
Fokker-Planck equation on a sample of observed cells ("discrete manifold"):

    T(x_i <- x_j) ~ V(x_i) * exp(-|x_i - x_j - tau v(x_j)|^2 / (2 h_j^2)),  columns sum to 1,

with v the drift learned by CC-Tempo (any mode), tau a macro time step and V(x_i) a sampling
volume correction (~ 1 / local density). Then
  * forward:  P_{k+1} = T_k P_k            -> simulated distributions, KL to data in bits
  * backward: h_k = T_k^T h_{k+1}          -> committors = fate probabilities, deterministic
  * stationary distribution of a frozen T  -> attractors, basins, effective potential -log pi
Differences from the PNAS method (documented in docs/DERIVATIONS.md): the drift is a neural
field instead of an interpolated potential, the kernel is restricted to k nearest neighbours
with an adaptive bandwidth (needed in 20-50 dimensions), and T is time-inhomogeneous because
the mean-field ligand field changes over time.
"""
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors


class PathIntegralOperator:
    def __init__(self, model, data, n_points=6000, dims=20, k=40, tau=0.5, vol_dim=5, seed=0,
                 force_include=None, device="cpu", kernel="mc", n_paths=32, dt=0.1, nodes="data",
                 sim_nodes=None):
        """force_include: dict {time_index: array of cell indices} that must be cloud nodes.

        kernel = 'mc' (default): column j of T is estimated by simulating n_paths short SDE paths of
                 length tau from node j and assigning the end points to their nearest node (a Markov
                 state model). Exact drift and noise; works in 20-50 dimensions.
        kernel = 'gaussian': the analytic short-time kernel of Cislo et al. (PNAS 2025). Only reliable
                 for low-dimensional, dense point clouds (it under-moves probability when tau*|v| is
                 smaller than the spacing between points).
        """
        self.model, self.data, self.dims, self.k, self.tau = model, data, dims, k, tau
        self.device, self.kernel, self.n_paths, self.dt, self.seed = device, kernel, n_paths, dt, seed
        rng = np.random.default_rng(seed)
        if nodes == "simulated":
            # Markov state model on the model's own states: sim_nodes = (n, d) tensor of simulated
            # states (e.g. a background population recorded along its trajectory); forced data cells
            # (progenitors) are appended so that they can be queried.
            extra = []
            if force_include:
                for i, idx in force_include.items():
                    extra += [(i, j) for j in np.asarray(idx)]
            sim = torch.as_tensor(sim_nodes).float().cpu()
            pick = rng.choice(sim.shape[0], min(n_points, sim.shape[0]), replace=False)
            self.node_time = np.array([-1] * len(pick) + [a for a, _ in extra])
            self.node_cell = np.array(list(pick) + [b for _, b in extra])
            self.X = torch.cat([sim[pick]] + ([torch.stack([data["x"][a][b] for a, b in extra])] if extra else []))
            self.N = self.X.shape[0]
            self.P = self.X[:, :dims].numpy()
            self.nn = NearestNeighbors(n_neighbors=k).fit(self.P)
            self.nn_full = NearestNeighbors(n_neighbors=1).fit(self.P)
            self.V = np.ones(self.N)
            self.celltype = np.array(["simulated"] * self.N, dtype=object)
            return
        T = len(data["x"])
        per = n_points // T
        nodes = []
        for i in range(T):
            n = data["x"][i].shape[0]
            pick = rng.choice(n, min(per, n), replace=False)
            if force_include and i in force_include:
                pick = np.union1d(pick, np.asarray(force_include[i]))
            nodes += [(i, j) for j in pick]
        self.node_time = np.array([a for a, _ in nodes])
        self.node_cell = np.array([b for _, b in nodes])
        self.X = torch.stack([data["x"][a][b] for a, b in nodes]).float()
        self.P = self.X[:, :dims].numpy()
        self.N = self.X.shape[0]
        self.nn = NearestNeighbors(n_neighbors=k).fit(self.P)
        self.nn_full = NearestNeighbors(n_neighbors=1).fit(self.P)
        r, _ = NearestNeighbors(n_neighbors=16).fit(self.P).kneighbors(self.P)
        rk = r[:, -1] / np.median(r[:, -1])
        self.V = np.clip(rk ** vol_dim, 0.1, 10.0)
        self.celltype = np.array([str(data["celltype"][a][b]) for a, b in nodes])

    def node_index(self, time_idx, cell_idx):
        lookup = {(a, b): n for n, (a, b) in enumerate(zip(self.node_time, self.node_cell))}
        return np.array([lookup[(time_idx, int(c))] for c in cell_idx])

    @torch.no_grad()
    def drift(self, field=None, batch=4096):
        out = []
        for s in range(0, self.N, batch):
            x = self.X[s:s + batch].to(self.device)
            y = None
            if self.model.mode == "v1":
                y = torch.zeros(x.shape[0], self.model.v1_dim, device=self.device)
            vx, _ = self.model.drift(x, y, field)
            out.append(vx.cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def _transition_mc(self, field=None, batch=2048):
        from ..simulate import simulate
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        n_steps_ft = None
        rows = []
        for s in range(0, self.N, batch):
            x = self.X[s:s + batch].to(self.device).repeat_interleave(self.n_paths, 0)
            y = torch.zeros(x.shape[0], self.model.v1_dim, device=self.device) if self.model.mode == "v1" else None
            ft = None
            if self.model.uses_field:
                n_steps_ft = max(1, int(round(self.tau / self.dt)))
                ft = (field if field is not None else torch.zeros(self.model.K, device=self.device))[None].expand(n_steps_ft, -1)
            out = simulate(self.model, x, 0.0, self.tau, self.dt, y0=y, field_traj=ft, generator=g)
            _, ind = self.nn_full.kneighbors(out["x"][:, :self.dims].cpu().numpy())
            rows.append(ind[:, 0])
        rows = np.concatenate(rows)
        cols = np.repeat(np.arange(self.N), self.n_paths)
        w = np.full(len(rows), 1.0 / self.n_paths)
        return sp.csr_matrix((w, (rows, cols)), shape=(self.N, self.N))

    def transition(self, field=None):
        if self.kernel == "mc":
            return self._transition_mc(field)
        v = self.drift(field)
        Y = (self.X.numpy() + self.tau * v)[:, :self.dims]
        dist, ind = self.nn.kneighbors(Y)
        base = (self.model.sigma ** 2) * self.tau
        h2 = np.maximum(base, (dist[:, self.k // 2] ** 2))[:, None]
        w = self.V[ind] * np.exp(-(dist ** 2 - dist[:, :1] ** 2) / (2 * h2))
        w = w / w.sum(1, keepdims=True)
        rows = ind.ravel()
        cols = np.repeat(np.arange(self.N), self.k)
        return sp.csr_matrix((w.ravel(), (rows, cols)), shape=(self.N, self.N))  # column j = from node j

    def transitions(self, t0, t1, dt, field_traj=None, exog_fn=None, alpha_fn=None):
        """List of macro-step operators from t0 to t1 (time-inhomogeneous if a field is used)."""
        n_macro = max(1, int(round((t1 - t0) / self.tau)))
        micro_per_macro = max(1, int(round(self.tau / dt)))
        Ts = []
        for m in range(n_macro):
            field = None
            t = t0 + m * self.tau
            if self.model.uses_field:
                k = min(m * micro_per_macro, (field_traj.shape[0] - 1) if field_traj is not None else 0)
                endo = field_traj[k].to(self.device) if field_traj is not None else torch.zeros(self.model.K, device=self.device)
                a = alpha_fn(t) if alpha_fn is not None else 1.0
                field = torch.as_tensor(a, device=self.device, dtype=torch.float32) * endo
                if exog_fn is not None:
                    field = field + exog_fn(t)
            Ts.append(self.transition(field))
        return Ts

    @staticmethod
    def evolve(Ts, p0):
        p = np.asarray(p0, float)
        out = [p]
        for T in Ts:
            p = T @ p
            out.append(p)
        return out

    @staticmethod
    def committor(Ts, terminal):
        h = np.asarray(terminal, float)
        for T in reversed(Ts):
            h = T.T @ h
        return h

    def histogram(self, X_obs, smooth=0):
        _, ind = NearestNeighbors(n_neighbors=1).fit(self.P).kneighbors(np.asarray(X_obs)[:, :self.dims])
        p = np.bincount(ind[:, 0], minlength=self.N).astype(float)
        p /= p.sum()
        return p


def kl_bits(p_obs, p_sim, eps=1e-10):
    p = np.asarray(p_obs) + eps
    q = np.asarray(p_sim) + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def entropy_bits(p, eps=1e-12):
    p = np.asarray(p) + eps
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def stationary(T, iters=3000, tol=1e-11, teleport=1e-4):
    N = T.shape[0]
    p = np.full(N, 1.0 / N)
    for _ in range(iters):
        q = (1 - teleport) * (T @ p) + teleport / N
        q /= q.sum()
        if np.abs(q - p).sum() < tol:
            return q
        p = q
    return p


def attractors(P, pi, k=15, min_mass=0.02):
    """Basins by steepest ascent of pi on a kNN graph; small basins are merged into neighbours."""
    _, ind = NearestNeighbors(n_neighbors=k).fit(P).kneighbors(P)
    parent = ind[np.arange(len(pi)), np.argmax(pi[ind], axis=1)]
    root = parent.copy()
    for _ in range(64):
        nxt = parent[root]
        if np.all(nxt == root):
            break
        root = nxt
    labels = root
    changed = True
    while changed:
        changed = False
        roots, inv = np.unique(labels, return_inverse=True)
        mass = np.bincount(inv, weights=pi)
        small = np.where(mass < min_mass)[0]
        if len(small) == 0 or len(roots) == 1:
            break
        s = small[np.argmin(mass[small])]
        members = np.where(inv == s)[0]
        nb = ind[members].ravel()
        nb = nb[inv[nb] != s]
        if len(nb) == 0:
            break
        target = roots[inv[nb[np.argmax(pi[nb])]]]
        labels[members] = target
        changed = True
    roots, inv = np.unique(labels, return_inverse=True)
    return roots, inv, np.bincount(inv, weights=pi)


@torch.no_grad()
def build_operator(model, data, t0, t1, clf, pi_cfg=None, dt=0.1, exog_fn=None, force_include=None,
                   device="cpu", n_bg=4000, seed=0, record_every=2):
    """Default operator: Markov state model on the model's own simulated states.

    A growth-weighted background population is simulated from t0 to t1 (with its mean field and
    exogenous ligands); its recorded states are the nodes. Each node is labelled by the cell-type
    classifier. Fates/committors computed on this operator agree with Monte-Carlo simulation
    (validated on the synthetic system), unlike nodes taken from the data cloud, onto which
    off-manifold model trajectories cannot be mapped faithfully.
    """
    from ..data.dataset import time_index
    from ..simulate import simulate
    pi_cfg = pi_cfg or {}
    i0 = time_index(data, t0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(data["x"][i0].shape[0], min(n_bg, data["x"][i0].shape[0]), replace=False)
    x = data["x"][i0][torch.from_numpy(idx)].to(device)
    g = data["growth"][i0][torch.from_numpy(idx)].to(device)
    y = data["v1"][i0][torch.from_numpy(idx)].to(device) if model.mode == "v1" else None
    gen = torch.Generator(device=device).manual_seed(seed)
    out = simulate(model, x, t0, t1, dt, y0=y, growth=g, exog_fn=exog_fn, generator=gen,
                   record_every=record_every, record_field=True)
    sim = torch.cat([x] + out["traj"]).cpu()
    op = PathIntegralOperator(model, data, int(pi_cfg.get("n_points", 8000)), int(pi_cfg.get("dims", 20)),
                              int(pi_cfg.get("k", 40)), float(pi_cfg.get("tau", 0.5)), seed=seed,
                              force_include=force_include, device=device, kernel="mc",
                              n_paths=int(pi_cfg.get("n_paths", 32)), dt=dt, nodes="simulated", sim_nodes=sim)
    op.celltype = np.array(clf.classes)[clf.predict_idx(op.X).numpy()]
    return op, out["fields"]
