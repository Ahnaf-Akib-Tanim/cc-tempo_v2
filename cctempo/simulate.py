"""Forward simulation (Euler-Maruyama) for all model modes, incl. the mean-field ligand field."""
import numpy as np
import torch


def _alpha_at(alpha_fn, t, device):
    if alpha_fn is None:
        return 1.0
    a = alpha_fn(t)
    return torch.as_tensor(a, dtype=torch.float32, device=device) if not np.isscalar(a) else float(a)


def simulate(model, x0, t0, t1, dt, y0=None, w0=None, growth=None, field_traj=None,
             exog_fn=None, alpha_fn=None, noise=True, create_graph=False, generator=None,
             record_every=0, record_field=False, field_weight_fn=None):
    """Simulate cells from time t0 to t1.

    model      : CCTempo
    x0         : (n, d) start states
    w0, growth : optional per-cell base weights and growth rates; the population weight of a
                 particle at time t is w0 * exp(growth * (t - t0)) (used for the ligand field).
    field_traj : optional precomputed endogenous field per step, (n_steps, K). When None and the
                 model uses a field, the field is computed from the simulated population itself.
    exog_fn    : t -> (K,) exogenous ligand field (cytokines, SAG, pulses ...).
    alpha_fn   : t -> scalar or (K,) multiplier on the endogenous field (communication strength,
                 temporal blocking windows).
    field_weight_fn : x -> (n,) multiplier on each particle's ligand contribution (sender
                 ablation, e.g. silence every cell currently classified as a monocyte).
    Returns dict(x=final x, y=final y, traj=list of recorded x, fields=(n_steps, K) tensor or None).
    """
    n_steps = int(round((t1 - t0) / dt))
    x, y = x0, y0
    dev = x0.device
    traj, fields = [], []
    if w0 is None:
        w0 = torch.ones(x0.shape[0], device=dev)
    for k in range(n_steps):
        t = t0 + k * dt
        field = None
        if model.uses_field:
            if field_traj is not None:
                endo = field_traj[k].to(dev)
            else:
                w = w0 * torch.exp(growth * (t - t0)) if growth is not None else w0
                if field_weight_fn is not None:
                    w = w * field_weight_fn(x.detach())
                # positions are detached: no 'sender force', but parameters (gates, doses) still get gradients
                endo = model.ligand_field(x.detach(), w)
            field = _alpha_at(alpha_fn, t, dev) * endo
            if exog_fn is not None:
                field = field + exog_fn(t)
            if record_field:
                fields.append(endo.detach())
        if noise:
            z = torch.randn(x.shape, device=dev, generator=generator) if generator is not None else torch.randn_like(x)
        else:
            z = torch.zeros_like(x)
        x, y = model.step(x, y, field, dt, z, create_graph=create_graph)
        if record_every and (k + 1) % record_every == 0:
            traj.append(x.detach())
    out = {"x": x, "y": y, "traj": traj, "fields": torch.stack(fields) if fields else None}
    return out


@torch.no_grad()
def background_field(model, x_bg, t0, t1, dt, growth=None, exog_fn=None, alpha_fn=None, seed=0,
                     field_weight_fn=None):
    """Simulate a background population and return its endogenous ligand field per step.

    Test particles (e.g. 2,000 copies of one progenitor) then feel this field without
    contributing to it, which is the mean-field limit of the interacting system.
    """
    if not model.uses_field:
        return None
    g = torch.Generator(device=x_bg.device).manual_seed(seed)
    out = simulate(model, x_bg, t0, t1, dt, growth=growth, exog_fn=exog_fn, alpha_fn=alpha_fn,
                   generator=g, record_field=True, field_weight_fn=field_weight_fn)
    return out["fields"]
