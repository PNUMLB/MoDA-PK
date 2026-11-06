"""Batched exact two-compartment solution, equivalent to reference_solver.

Events are [start hours, amount umol, route (0 oral/1 bolus/2 infusion),
duration hours]. Zero-amount padded events contribute nothing. This solver
changes numerical evaluation only; it does not change the compartment ODE.
"""
import torch


def exprel_negative(x):
    """(1-exp(-x))/x for x >= 0, with differentiable series at zero."""
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    series = 1 - x / 2 + x.square() / 6 - x.pow(3) / 24 + x.pow(4) / 120
    return torch.where(small, series, -torch.expm1(-x) / safe)


def curve_single(log_parameters, events, times):
    ka, cl, v1, q, v2 = log_parameters.exp().unbind(-1)
    k10, k12, k21 = cl / v1, q / v1, q / v2
    total = k10 + k12 + k21
    gap = torch.sqrt((k10 + k12 - k21).square() + 4 * k12 * k21)
    alpha = (total + gap) / 2
    # Product form avoids cancellation for a small terminal elimination rate.
    beta = k10 * k21 / alpha
    a = (alpha - k21) / gap
    b = (k21 - beta) / gap
    dt = (times[:, None] - events[None, :, 0]).clamp_min(0)
    amount, route, duration = events[:, 1], events[:, 2], events[:, 3]

    def oral_component(rate):
        difference = (rate - ka).abs()
        return dt * torch.exp(-torch.minimum(rate, ka) * dt) * exprel_negative(difference * dt)

    oral = ka * (a * oral_component(alpha) + b * oral_component(beta))
    bolus = a * torch.exp(-alpha * dt) + b * torch.exp(-beta * dt)
    on = torch.minimum(dt, duration[None, :])
    after = (dt - duration[None, :]).clamp_min(0)
    infusion = on * (a * exprel_negative(alpha * on) * torch.exp(-alpha * after)
                     + b * exprel_negative(beta * on) * torch.exp(-beta * after))
    infusion = infusion / duration.clamp_min(1e-12)[None, :]
    response = torch.where(route[None, :] == 0, oral,
                           torch.where(route[None, :] == 1, bolus, infusion))
    response = response * (times[:, None] >= events[None, :, 0])
    concentration = (response * amount[None, :]).sum(-1) / v1
    return torch.log(concentration.clamp_min(0) + 1e-5)


batch_curve = torch.vmap(curve_single)
batch_jacobian = torch.vmap(torch.func.jacfwd(curve_single, argnums=0))


def pack_events(events, *, dtype=torch.float64):
    return torch.tensor([[e.time, e.amount, {'oral': 0, 'iv_bolus': 1, 'iv_infusion': 2}[e.route],
                          e.duration] for e in events], dtype=dtype)
