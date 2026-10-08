"""Move between PCA state space and (scaled) gene space; gene-level in-silico perturbations."""
import numpy as np
import torch


def gene_index(data, genes, strict=False):
    pos = {g: i for i, g in enumerate(data["genes"])}
    idx, missing = [], []
    for g in genes:
        (idx.append(pos[g]) if g in pos else missing.append(g))
    if strict and missing:
        raise KeyError(f"genes not in dataset: {missing}")
    return np.array(idx, dtype=int), missing


def decode_scaled(x, data):
    C = torch.as_tensor(np.asarray(data["pca"]["components"]), dtype=torch.float32, device=x.device)
    m = torch.as_tensor(np.asarray(data["pca"]["mean"]), dtype=torch.float32, device=x.device)
    return x @ C + m


def encode_scaled(g, data):
    C = torch.as_tensor(np.asarray(data["pca"]["components"]), dtype=torch.float32, device=g.device)
    m = torch.as_tensor(np.asarray(data["pca"]["mean"]), dtype=torch.float32, device=g.device)
    return (g - m) @ C.T


def perturb_genes(x, data, genes, value, mode="set"):
    """PRESCIENT-style perturbation: set (or shift) the scaled z-score of genes, re-project to PCA.

    Only the component of the perturbation inside the PCA subspace survives; we report its
    retained norm so readers can judge how well a gene is represented (a limitation of every
    PCA-based model, incl. PRESCIENT and CC-Tempo v1).
    """
    idx, missing = gene_index(data, genes)
    g = decode_scaled(x, data)
    g2 = g.clone()
    if mode == "set":
        g2[:, idx] = float(value)
    else:
        g2[:, idx] = g2[:, idx] + float(value)
    delta = g2 - g
    x2 = x + encode_scaled(delta + torch.as_tensor(np.asarray(data["pca"]["mean"]), dtype=torch.float32, device=x.device), data)
    retained = float((encode_scaled(delta + torch.as_tensor(np.asarray(data["pca"]["mean"]), dtype=torch.float32, device=x.device), data).norm(dim=1)
                      / delta.norm(dim=1).clamp_min(1e-8)).mean())
    return x2, {"missing": missing, "retained_fraction": retained}
