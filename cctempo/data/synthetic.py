"""Synthetic ground truth: a fate decision whose outcome depends on cell-cell communication.

Latent dynamics (2-D), the heteroclinic-flip landscape of Cislo et al. (PNAS 2025, Eq. 7):
    U(x, y) = x^4 + y^4 + x^3 - 2 x y^2 - x^2 + a x + b_i y
with a cell-specific tilt   b_i = b0 + beta * (L(t) + e) * r_i,
    L(t) = mean_j l_j        (ligand secreted by cells already in the upper daughter valley)
    r_i  = receptor level of cell i (high in progenitors, low after commitment)
    e    = exogenous ligand dose (the experimental 'condition').
beta > 0: negative feedback -- the more cells commit to the upper fate, the more ligand, the
more the remaining progenitors are tilted towards the lower fate (population-level balancing).
beta = 0: cells are autonomous (control: the model must not invent communication).

The latent state is embedded into G 'genes' (nonlinear programme genes + the true ligand and
receptor genes + decoy ligand/receptor genes that carry no causal signal), giving data with
exactly the structure CC-Tempo v2 assumes, but with a known answer.
"""
import numpy as np
import torch

from .dataset import make_split


def _grad_U(x, y, a, b):
    gx = 4 * x ** 3 + 3 * x ** 2 - 2 * y ** 2 - 2 * x + a
    gy = 4 * y ** 3 - 4 * x * y + b
    return gx, gy


def ligand_level(x, y):
    return 1.0 / (1.0 + np.exp(-6 * (y - 0.4))) * (1.0 / (1.0 + np.exp(-6 * (x - 0.3))))


def receptor_level(x, y):
    return 1.0 / (1.0 + np.exp(6 * (x - 0.0)))


def simulate_latent(x0, y0, t_obs, beta, e, a=-1.4, b0=0.0, D=0.08, dt=0.01, seed=0, field=None):
    """Simulate N interacting cells; returns dict t -> (x, y) snapshots and the field trace."""
    rng = np.random.default_rng(seed)
    x, y = x0.copy(), y0.copy()
    out, t, trace = {}, 0.0, []
    for t_next in t_obs:
        while t < t_next - 1e-9:
            L = ligand_level(x, y).mean() if field is None else field(t)
            trace.append(L)
            b = b0 + beta * (L + e) * receptor_level(x, y)
            gx, gy = _grad_U(x, y, a, b)
            x = x - gx * dt + np.sqrt(2 * D * dt) * rng.standard_normal(x.shape)
            y = y - gy * dt + np.sqrt(2 * D * dt) * rng.standard_normal(y.shape)
            t += dt
        out[t_next] = (x.copy(), y.copy())
    return out, np.array(trace)


def latent_label(x, y):
    lab = np.where(x < 0.0, "Prog", np.where(y > 0, "A", "B"))
    return lab


def _embed(x, y, rng_state, n_prog=40, n_decoy=5, noise=0.1):
    rng = np.random.default_rng(rng_state)
    feats = np.stack([x, y, x ** 2, y ** 2, x * y], 1)
    A = rng.normal(0, 1, (5, n_prog))
    genes = np.tanh(feats @ A / 2.0) + 1.0
    lig = 2 * ligand_level(x, y)
    rec = 2 * receptor_level(x, y)
    decoys = []
    for d in range(2 * n_decoy):
        w = rng.normal(0, 1, 5)
        decoys.append(1 + np.tanh(feats @ w / 2))
    X = np.column_stack([genes, lig, rec] + decoys)
    X = X + noise * np.random.default_rng(rng_state + 1).standard_normal(X.shape)
    X = np.clip(X, 0, None)
    names = [f"G{i}" for i in range(n_prog)] + ["LigA", "RecA"] + \
            [f"LigD{i}" for i in range(n_decoy)] + [f"RecD{i}" for i in range(n_decoy)]
    return X.astype(np.float32), names


def make_synthetic(beta=1.5, doses=(0.0, 0.5, 1.0), n_cells=3000, t_obs=(1.5, 3.0), a=-1.4, D=0.08,
                   n_pcs=10, seed=0, embed_seed=7):
    """Build a data.pt-style dict with conditions 'e<dose>' and ground-truth fate probabilities."""
    rng = np.random.default_rng(seed)
    x0 = -1.0 + 0.15 * rng.standard_normal(n_cells)
    y0 = 0.15 * rng.standard_normal(n_cells)
    snaps = {0.0: {"start": (x0, y0)}}
    traces = {}
    for e in doses:
        out, tr = simulate_latent(x0, y0, list(t_obs), beta, e, a=a, D=D, seed=seed + int(100 * e) + 1)
        traces[e] = tr
        for t in t_obs:
            snaps.setdefault(t, {})[f"e{e}"] = out[t]
    # assemble cells
    times = [0.0] + list(t_obs)
    lat, cond, tim = [], [], []
    for t in times:
        for c, (xx, yy) in snaps[t].items():
            lat.append(np.stack([xx, yy], 1)); cond += [c] * len(xx); tim += [t] * len(xx)
    lat = np.concatenate(lat)
    X, names = _embed(lat[:, 0], lat[:, 1], embed_seed)
    Xl = np.log1p(X)
    mu, sd = Xl.mean(0), Xl.std(0)
    sd[sd == 0] = 1
    Xs = (Xl - mu) / sd
    from sklearn.decomposition import PCA
    pca = PCA(n_pcs, random_state=0).fit(Xs)
    Z = pca.transform(Xs).astype(np.float32)
    tim, cond = np.array(tim), np.array(cond)
    labels = latent_label(lat[:, 0], lat[:, 1])
    data = {"x": [], "times": times, "growth": [], "celltype": [], "split": [], "obs_index": [], "condition": [],
            "v1": None, "genes": names}
    for t in times:
        idx = np.where(tim == t)[0]
        data["x"].append(torch.from_numpy(Z[idx])); data["growth"].append(torch.zeros(len(idx)))
        data["celltype"].append(labels[idx]); data["split"].append(make_split(len(idx), 0.1, seed))
        data["obs_index"].append(idx); data["condition"].append(cond[idx])
    data["pca"] = {"components": pca.components_.astype(np.float32), "mean": pca.mean_.astype(np.float32)}
    data["scaler"] = {"mean": mu.astype(np.float32), "scale": sd.astype(np.float32)}
    n_decoy = sum(1 for g in names if g.startswith("LigD"))
    data["lr"] = {"ligand": ["LigA"] + [f"LigD{i}" for i in range(n_decoy)],
                  "receptor": ["RecA"] + [f"RecD{i}" for i in range(n_decoy)],
                  "lig_genes": [["LigA"]] + [[f"LigD{i}"] for i in range(n_decoy)],
                  "rec_genes": [["RecA"]] + [[f"RecD{i}"] for i in range(n_decoy)],
                  "prior": [0.5] * (1 + n_decoy), "family": ["TRUE"] + [f"DECOY{i}" for i in range(n_decoy)],
                  "exogenous": ["EXO"] + [None] * n_decoy}
    data["meta"] = {"name": "synthetic", "beta": beta, "doses": list(doses), "a": a, "D": D, "t_obs": list(t_obs)}
    # ground-truth fate probability P(A | start cell, dose) by re-simulating copies in the true field
    gt = {}
    sub = rng.choice(n_cells, 300, replace=False)
    for e in doses:
        tr = traces[e]
        field = lambda t, tr=tr: tr[min(int(round(t / 0.01)), len(tr) - 1)]
        reps = 100
        xs, ys = np.repeat(x0[sub], reps), np.repeat(y0[sub], reps)
        out, _ = simulate_latent(xs, ys, [t_obs[-1]], beta, e, a=a, D=D, seed=seed + 999, field=field)
        lab = latent_label(*out[t_obs[-1]]).reshape(len(sub), reps)
        pa, pb = (lab == "A").sum(1), (lab == "B").sum(1)
        gt[f"e{e}"] = {"prog_idx": sub, "p_A": (pa + 1) / (pa + pb + 2),
                       "frac": {k: float((lab == k).mean()) for k in ("A", "B", "Prog")}}
    data["eval"] = {"synthetic": gt}
    return data
