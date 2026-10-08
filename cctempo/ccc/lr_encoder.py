"""Differentiable ligand / receptor read-out from the model state.

The model state x lives in PCA space of scaled log-expression. For every ligand-receptor
(LR) pair k we decode only the genes we need:

    log-expr_g(x) = (x @ W[:, g] + m_pca[g]) * s[g] + m[g]

and form complex expression with a soft-min over subunits (LIANA/CellPhoneDB use the
minimum subunit for heteromeric complexes). Everything is linear + smooth, so gradients
flow from the received signal back to the cell state.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _pad_index(groups, lookup):
    width = max(1, max(len(g) for g in groups))
    idx = np.full((len(groups), width), -1, dtype=np.int64)
    for k, g in enumerate(groups):
        for j, gene in enumerate(g):
            idx[k, j] = lookup[gene]
    return idx


class LREncoder(nn.Module):
    def __init__(self, components, pca_mean, sc_mean, sc_scale, genes, lig_genes, rec_genes,
                 softmin_tau=0.1, softplus_beta=10.0):
        """
        components: (d, G) PCA loadings; pca_mean: (G,); sc_mean/sc_scale: (G,) scaler stats
        genes: list of G gene names; lig_genes/rec_genes: list (len K) of lists of subunit genes.
        """
        super().__init__()
        gene_pos = {g: i for i, g in enumerate(genes)}
        for grp in list(lig_genes) + list(rec_genes):
            for g in grp:
                if g not in gene_pos:
                    raise KeyError(f"LR gene {g} is not in the dataset gene list")
        needed = sorted({gene_pos[g] for grp in list(lig_genes) + list(rec_genes) for g in grp})
        local = {genes[i]: j for j, i in enumerate(needed)}
        comp = np.asarray(components, dtype=np.float32)[:, needed]
        scale = np.asarray(sc_scale, dtype=np.float32)[needed]
        offset = (np.asarray(pca_mean, dtype=np.float32)[needed] * scale
                  + np.asarray(sc_mean, dtype=np.float32)[needed])
        self.register_buffer("W", torch.from_numpy(comp * scale[None, :]))
        self.register_buffer("b", torch.from_numpy(offset))
        self.register_buffer("lig_idx", torch.from_numpy(_pad_index(lig_genes, local)))
        self.register_buffer("rec_idx", torch.from_numpy(_pad_index(rec_genes, local)))
        self.K = len(lig_genes)
        self.tau = softmin_tau
        self.beta = softplus_beta
        self.needed_genes = [genes[i] for i in needed]

    def gene_expr(self, x):
        """Nonnegative decoded log-expression of the needed genes, (n, g)."""
        return F.softplus(x @ self.W + self.b, beta=self.beta)

    def _complex(self, e, idx):
        mask = idx >= 0
        safe = idx.clamp_min(0)
        vals = e[:, safe]  # (n, K, width)
        vals = vals.masked_fill(~mask[None], float("inf"))
        return -self.tau * torch.logsumexp(-vals / self.tau, dim=-1)

    def forward(self, x):
        e = self.gene_expr(x)
        return self._complex(e, self.lig_idx), self._complex(e, self.rec_idx)

    def ligand(self, x):
        return self._complex(self.gene_expr(x), self.lig_idx)

    def receptor(self, x):
        return self._complex(self.gene_expr(x), self.rec_idx)
