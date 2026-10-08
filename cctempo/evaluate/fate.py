"""Monte-Carlo fate prediction: simulate copies of progenitors and classify final states."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ..simulate import background_field, simulate


class CellTypeClassifier:
    """Multinomial logistic regression on PCA coordinates (deliberately simple, as in the paper)."""

    def __init__(self, max_cells=60000, seed=0, C=1.0):
        self.max_cells, self.seed, self.C = max_cells, seed, C

    def fit(self, X, labels):
        X, labels = np.asarray(X), np.asarray(labels).astype(str)
        rng = np.random.default_rng(self.seed)
        if X.shape[0] > self.max_cells:
            idx = rng.choice(X.shape[0], self.max_cells, replace=False)
            X, labels = X[idx], labels[idx]
        self.lr = LogisticRegression(max_iter=1000, C=self.C)
        self.lr.fit(X, labels)
        self.classes = list(self.lr.classes_)
        W = self.lr.coef_
        b = self.lr.intercept_
        if W.shape[0] == 1:  # binary sklearn convention
            W = np.vstack([-W, W])
            b = np.array([-b[0], b[0]])
        self.W = torch.tensor(W, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        return self

    def logits(self, x):
        return x @ self.W.to(x.device).T + self.b.to(x.device)

    def predict_idx(self, x):
        return self.logits(x).argmax(1)

    def predict_proba(self, X):
        return self.lr.predict_proba(np.asarray(X))

    def index(self, name):
        return self.classes.index(name)


def fit_classifier(data, seed=0, max_cells=60000):
    X = torch.cat(data["x"], 0).numpy()
    y = np.concatenate([np.asarray(c).astype(str) for c in data["celltype"]])
    return CellTypeClassifier(max_cells=max_cells, seed=seed).fit(X, y)


@torch.no_grad()
def mc_fate_counts(model, x_prog, t0, t1, dt, clf, n_copies=2000, max_particles=200_000,
                   field_traj=None, exog_fn=None, alpha_fn=None, y_prog=None, seed=0, device=None):
    """Return (n_prog, n_classes) counts of predicted final cell types.

    Noise is drawn from a generator seeded per chunk, so two calls with the same seed use
    common random numbers -> perturbation effects are measured as paired differences.
    """
    device = device or x_prog.device
    n_prog = x_prog.shape[0]
    per_chunk = max(1, max_particles // n_copies)
    C = len(clf.classes)
    counts = torch.zeros(n_prog, C, dtype=torch.long)
    for c0 in range(0, n_prog, per_chunk):
        c1 = min(n_prog, c0 + per_chunk)
        x = x_prog[c0:c1].to(device).repeat_interleave(n_copies, 0)
        y = y_prog[c0:c1].to(device).repeat_interleave(n_copies, 0) if y_prog is not None else None
        g = torch.Generator(device=device).manual_seed(int(seed) * 100003 + c0)
        out = simulate(model, x, t0, t1, dt, y0=y, field_traj=field_traj, exog_fn=exog_fn,
                       alpha_fn=alpha_fn, generator=g)
        pred = clf.predict_idx(out["x"]).view(c1 - c0, n_copies)
        for j in range(C):
            counts[c0:c1, j] = (pred == j).sum(1).cpu()
    return counts.numpy()


def fate_bias_from_counts(counts, clf, pos="Neutrophil", neg="Monocyte", pseudo=1.0):
    p = counts[:, clf.index(pos)].astype(float)
    n = counts[:, clf.index(neg)].astype(float)
    bias = (p + pseudo) / (p + n + 2 * pseudo)
    mask = (p + n) > 0
    return bias, mask


def composition(counts, clf):
    tot = counts.sum(0).astype(float)
    return dict(zip(clf.classes, (tot / tot.sum()).tolist()))


def prepare_field(model, data, t0, t1, dt, n_bg=5000, exog_fn=None, alpha_fn=None, seed=0,
                  device="cpu", bg_mask=None, condition=None, field_weight_fn=None):
    """Simulate a growth-weighted background population from the start time; returns field traj."""
    if not model.uses_field:
        return None
    from ..data.dataset import cells, time_index
    i = time_index(data, t0)
    m = cells(data, t0, "all", condition) if condition is not None else np.ones(data["x"][i].shape[0], bool)
    if bg_mask is not None:
        m &= bg_mask
    idx = np.where(m)[0]
    rng = np.random.default_rng(seed)
    if len(idx) > n_bg:
        idx = rng.choice(idx, n_bg, replace=False)
    x_bg = data["x"][i][idx].to(device)
    g_bg = data["growth"][i][idx].to(device)
    return background_field(model, x_bg, t0, t1, dt, growth=g_bg, exog_fn=exog_fn, alpha_fn=alpha_fn, seed=seed,
                            field_weight_fn=field_weight_fn)
