"""Gene-level attributions.

1. potential attribution (the original paper's analysis): Integrated Gradients of the landscape
   value U at a state, w.r.t. scaled gene expression (PCA is a fixed linear layer).
2. fate attribution (new): Integrated Gradients of the fate logit (pos - neg class) of the
   *simulated end state* w.r.t. the *initial* gene expression, differentiating through the SDE
   with frozen noise. This asks "which progenitor genes push the eventual fate", which is what
   the paper's Day-2 analysis intended.
captum is used when available; otherwise a plain Riemann IG is computed.
"""
import numpy as np
import torch

from ..geneops import decode_scaled, encode_scaled
from ..simulate import simulate


def _ig(f, inputs, baseline, steps=32):
    total = torch.zeros_like(inputs)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        z = (baseline + a * (inputs - baseline)).detach().requires_grad_(True)
        out = f(z).sum()
        total += torch.autograd.grad(out, z)[0]
    return (inputs - baseline) * total / steps


def potential_attribution(model, data, x, field=None, steps=32):
    """IG of U(x) w.r.t. scaled genes; returns (n, G) attributions."""
    g = decode_scaled(x, data).detach()
    base = torch.as_tensor(np.asarray(data["pca"]["mean"]), dtype=torch.float32, device=x.device)[None].expand_as(g)

    def f(gz):
        xx = encode_scaled(gz, data)
        s = None
        if model.mode == "local":
            s = model.signal(xx)
        elif model.mode == "meanfield":
            s = model.signal(xx, field)
        y = torch.zeros(xx.shape[0], model.v1_dim, device=xx.device) if model.mode == "v1" else None
        return model.potential(xx, y, s)

    return _ig(f, g, base, steps).detach()


def fate_attribution(model, data, clf, x0, t0, t1, dt, pos, neg, field_traj=None, exog_fn=None,
                     steps=16, n_noise=4, seed=0):
    """IG of the simulated end-state fate logit w.r.t. initial scaled genes, (n, G)."""
    g0 = decode_scaled(x0, data).detach()
    base = torch.as_tensor(np.asarray(data["pca"]["mean"]), dtype=torch.float32, device=x0.device)[None].expand_as(g0)
    ip, ineg = clf.index(pos), clf.index(neg)

    def f(gz):
        xx = encode_scaled(gz, data)
        acc = 0.0
        for r in range(n_noise):
            gen = torch.Generator(device=xx.device).manual_seed(seed + r)
            y = None
            if model.mode == "v1":
                y = torch.zeros(xx.shape[0], model.v1_dim, device=xx.device)
            out = simulate(model, xx, t0, t1, dt, y0=y, field_traj=field_traj, exog_fn=exog_fn,
                           create_graph=True, generator=gen)
            lg = clf.logits(out["x"])
            acc = acc + (lg[:, ip] - lg[:, ineg])
        return acc / n_noise

    return _ig(f, g0, base, steps).detach()


def rank_genes(attr, genes, top=20):
    a = attr.cpu().numpy()
    score = np.abs(a).mean(0)
    sign = np.sign(a.mean(0))
    order = np.argsort(-score)[:top]
    return [(genes[i], float(score[i]), int(sign[i])) for i in order]
