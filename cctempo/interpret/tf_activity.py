"""Signal -> transcription factor link (Reviewer 1(a): 'dynamics depend on key TFs').

TF activity of a (decoded) cell state = weighted mean of its CollecTRI targets' scaled
expression (a linear, univariate-model-like score). Comparing simulated end states with the
communication channel ON vs OFF (same random numbers) shows which TF programmes a ligand
acts through, i.e. LR -> TF -> fate.
"""
import numpy as np
import pandas as pd
import torch

from ..geneops import decode_scaled
from ..simulate import simulate


def load_collectri(organism="mouse"):
    import decoupler as dc
    for getter in (lambda: dc.op.collectri(organism=organism),            # decoupler >= 2
                   lambda: dc.get_collectri(organism=organism, split_complexes=False)):  # decoupler 1.x
        try:
            net = getter()
            break
        except Exception:
            net = None
    if net is None:
        raise RuntimeError("could not load CollecTRI via decoupler (needs internet)")
    return net.rename(columns={"weight": "mor"}) if "weight" in net.columns else net


def tf_matrix(net, genes, min_targets=5):
    pos = {g: i for i, g in enumerate(genes)}
    net = net[net["target"].isin(pos)]
    tfs = [tf for tf, sub in net.groupby("source") if len(sub) >= min_targets]
    W = np.zeros((len(tfs), len(genes)), dtype=np.float32)
    for r, tf in enumerate(tfs):
        sub = net[net["source"] == tf]
        for tgt, w in zip(sub["target"], sub["mor"]):
            W[r, pos[tgt]] = w
        W[r] /= max(1.0, np.abs(W[r]).sum())
    return tfs, torch.from_numpy(W)


@torch.no_grad()
def signal_tf_effect(model, data, x_prog, t0, t1, dt, field_on, field_off, tfs, W, n_copies=50, seed=0):
    """Mean TF-activity difference (ON - OFF) of simulated end states and a paired t-score."""
    dev = x_prog.device
    x = x_prog.repeat_interleave(n_copies, 0)
    res = []
    for ft in (field_on, field_off):
        g = torch.Generator(device=dev).manual_seed(seed)
        out = simulate(model, x, t0, t1, dt, field_traj=ft, generator=g)
        res.append(decode_scaled(out["x"], data) @ W.to(dev).T)
    d = (res[0] - res[1]).view(x_prog.shape[0], n_copies, -1).mean(1)
    mean = d.mean(0)
    se = d.std(0) / np.sqrt(d.shape[0])
    df = pd.DataFrame({"tf": tfs, "delta_activity": mean.cpu().numpy(), "t": (mean / se.clamp_min(1e-8)).cpu().numpy()})
    return df.sort_values("t", key=np.abs, ascending=False).reset_index(drop=True)
