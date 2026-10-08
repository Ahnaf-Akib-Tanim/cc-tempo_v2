"""Landscape geometry from the learned dynamics: attractors, basins and a bifurcation diagram
as a function of communication strength alpha (the endogenous analogue of the morphogen dose in
Cislo et al., PNAS 2025)."""
import numpy as np
import pandas as pd
import torch

from ..evaluate.pathint import attractors, stationary


def basin_table(op, pi, labels, roots):
    rows = []
    for b, r in enumerate(roots):
        members = labels == b
        w = pi[members]
        types, inv = np.unique(op.celltype[members], return_inverse=True)
        comp = np.bincount(inv, weights=w)
        rows.append({"basin": b, "mass": float(w.sum()), "majority_type": types[np.argmax(comp)],
                     "majority_share": float(comp.max() / comp.sum())})
    return pd.DataFrame(rows)


def communication_bifurcation(model, op, field_end, alphas, exog=None, min_mass=0.02):
    """Sweep alpha * endogenous field; count attractors and record basin masses per cell type."""
    rows, details = [], {}
    for a in alphas:
        field = None
        if model.uses_field:
            field = float(a) * field_end
            if exog is not None:
                field = field + exog
        T = op.transition(field)
        pi = stationary(T)
        roots, labels, mass = attractors(op.P, pi, min_mass=min_mass)
        bt = basin_table(op, pi, labels, roots)
        details[float(a)] = {"basins": bt, "U_eff": -np.log(pi + 1e-12)}
        row = {"alpha": float(a), "n_attractors": len(roots)}
        for t, m in bt.groupby("majority_type")["mass"].sum().items():
            row[f"mass_{t}"] = float(m)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0), details
