"""OT-Conditional Flow Matching baseline (Tong et al. 2024), self-contained (no torchcfm needed).

A time-dependent velocity field v(x, t) is regressed on straight paths between minibatch-OT
paired cells of consecutive time points. Fate: integrate progenitors to the final time, use the
classifier's soft probabilities. Optional SDE noise gives an SF2M-like stochastic variant.
"""
import os

os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")

import numpy as np
import torch
from torch import nn


class VelocityNet(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, d))

    def forward(self, x, t):
        """x: (n, d); t: (n, 1) times."""
        return self.net(torch.cat([x, t], 1))


def _ot_pairs(x0, x1):
    import ot
    M = torch.cdist(x0, x1).pow(2).detach().cpu().numpy()
    G = ot.emd(np.full(len(x0), 1 / len(x0)), np.full(len(x1), 1 / len(x1)), M)
    p = G.ravel() / G.sum()
    idx = np.random.choice(p.size, len(x0), p=p)
    i, j = np.unravel_index(idx, G.shape)
    return x0[i], x1[j]


def train_cfm(X_by_time, times, n_iter=20000, batch=256, lr=1e-3, sigma=0.1, device="cpu", seed=0, log_every=2000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = X_by_time[0].shape[1]
    model = VelocityNet(d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xs = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in X_by_time]
    for it in range(n_iter):
        k = np.random.randint(len(times) - 1)
        a, b = Xs[k], Xs[k + 1]
        x0 = a[torch.randint(len(a), (batch,), device=device)]
        x1 = b[torch.randint(len(b), (batch,), device=device)]
        x0, x1 = _ot_pairs(x0, x1)
        s = torch.rand(batch, 1, device=device)
        span = float(times[k + 1] - times[k])
        xt = (1 - s) * x0 + s * x1 + sigma * torch.randn_like(x0)
        t = times[k] + s * span
        u = (x1 - x0) / span
        loss = ((model.net(torch.cat([xt, t], 1)) - u) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (it + 1) % log_every == 0:
            print(f"[cfm] iter {it+1} loss {float(loss):.4f}")
    return model


@torch.no_grad()
def integrate(model, x0, t0, t1, dt=0.05, sde_sigma=0.0, seed=0):
    g = torch.Generator(device=x0.device).manual_seed(seed)
    x = x0.clone()
    n = int(round((t1 - t0) / dt))
    for k in range(n):
        t = torch.full((x.shape[0], 1), t0 + k * dt, device=x.device)
        x = x + model.net(torch.cat([x, t], 1)) * dt
        if sde_sigma > 0:
            x = x + sde_sigma * (dt ** 0.5) * torch.randn(x.shape, device=x.device, generator=g)
    return x
