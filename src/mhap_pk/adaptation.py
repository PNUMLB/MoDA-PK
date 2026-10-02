"""Restricted MAP adaptations used in the E2 component study."""
import torch
from .relational import floor_curve


def restricted_map(model, z, events, support_t, support_y, support_mask,
                   drug_indices=None, mode='all', steps=50, lr=.01, prior_scale=.6):
    """Fit only the requested prior deviation using support labels.

    ``amplitude`` is an additive log-concentration calibration with a scalar
    zero-centred prior. ``cl``, ``cl_vc`` and ``all`` optimize log PK
    parameters.  For an IV-only profile, ka is automatically frozen in the
    ``all`` condition because it is not identified by an IV input.
    """
    with torch.no_grad():
        mu, _, _ = model.prior(z, drug_indices)
        if mode == 'amplitude':
            start = mu.new_zeros((len(z), 1))
        else:
            start = mu.new_zeros(mu.shape)
    if mode == 'amplitude':
        active = torch.ones_like(start)
    else:
        active = torch.zeros_like(start)
        if mode == 'cl': active[:, 1] = 1
        elif mode == 'cl_vc': active[:, 1:3] = 1
        elif mode == 'all': active[:] = 1
        else: raise ValueError(mode)
        # No oral event means ka cannot be learned from the route history.
        iv_only = (events[..., 2] != 0).all(1)
        active[iv_only, 0] = 0
    u = start.clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    def objective(value):
        if mode == 'amplitude':
            curve = floor_curve(mu, events, support_t) + value
            penalty = (value[:, 0] / prior_scale).square()
        else:
            curve = floor_curve(mu + value * active, events, support_t)
            penalty = (value * active / prior_scale).square().sum(1)
        noise = getattr(model, 'noise', torch.as_tensor(.3, dtype=mu.dtype, device=mu.device))
        return .5 * ((((curve - support_y) / noise) ** 2) * support_mask).sum(1) + .5 * penalty
    with torch.no_grad():
        best_u = start.detach().clone(); best_obj = objective(start).detach()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        obj = objective(u)
        loss = obj.sum()
        if not torch.isfinite(loss): break
        loss.backward(); torch.nn.utils.clip_grad_norm_([u], 5.); opt.step()
        with torch.no_grad():
            better = torch.isfinite(obj) & (obj < best_obj)
            best_u = torch.where(better[:, None], u, best_u); best_obj = torch.minimum(best_obj, obj)
    if mode == 'amplitude':
        theta = mu; offset = best_u
    else:
        theta = mu + best_u * active; offset = torch.zeros((len(z), 1), dtype=mu.dtype, device=mu.device)
    return dict(parameters=theta.detach(), amplitude=offset.detach(), prior_parameters=mu.detach(),
                objective=best_obj.detach(), active=active.detach(), mode=mode)


def restricted_forecast(conditioned, model, events, query_t):
    curve = floor_curve(conditioned['parameters'], events, query_t)
    return curve + conditioned['amplitude']
