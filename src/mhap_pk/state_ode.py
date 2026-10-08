"""Exact amount-state propagation for oral, bolus and ongoing infusion events.

State order: depot, central, peripheral (umol). Parameters: ka, CL, Vc, Q, Vp.
The start state is post-event except when include_start=True. An infusion
crossing a filtering boundary delivers only its remaining portion afterward.
"""
import torch
from .kinetics import exprel_negative


def convolution(rate, ka, dt):
    """Integral exp(-rate*(t-s)) exp(-ka*s) ds, stable at rate == ka."""
    difference = (rate - ka) * dt
    small = difference.abs() < 1e-4
    # Symmetric series preserves derivatives at coincident rates.
    value = dt * torch.exp(-torch.minimum(rate, ka) * dt) * exprel_negative(difference.abs())
    series = dt * torch.exp(-.5 * (rate + ka) * dt) * (1 + difference.square()/24 + difference.pow(4)/1920)
    return torch.where(small, series, value)


def coefficients(log_parameters, extra_dims):
    params = log_parameters.exp()
    ka, cl, vc, q, vp = [v.reshape((-1,) + (1,)*extra_dims) for v in params.unbind(-1)]
    k10, k12, k21 = cl/vc, q/vc, q/vp
    gap = ((k10+k12-k21).square() + 4*k12*k21).sqrt()
    alpha = (k10+k12+k21+gap)/2
    beta = k10*k21/alpha
    a, b = (alpha-k21)/gap, (k21-beta)/gap
    return ka, k12, k21, alpha, beta, gap, a, b


def transition(log_parameters, dt):
    """Nonnegative 3x3 amount transition matrices for dt[B,...] >= 0."""
    ka, k12, k21, alpha, beta, gap, a, b = coefficients(log_parameters, dt.ndim-1)
    ea, eb, eg = torch.exp(-alpha*dt), torch.exp(-beta*dt), torch.exp(-ka*dt)
    cross = dt * eb * exprel_negative(gap*dt)
    fa, fb = convolution(alpha, ka, dt), convolution(beta, ka, dt)
    zero = torch.zeros_like(dt)
    depot_to_c = ka*(a*fa+b*fb)
    depot_to_p = (ka*k12/gap*(fb-fa)).clamp_min(0)
    return torch.stack([
        torch.stack([eg, zero, zero], -1),
        torch.stack([depot_to_c, a*ea+b*eb, k21*cross], -1),
        torch.stack([depot_to_p, k12*cross, b*ea+a*eb], -1),
    ], -2)


def infusion_response(log_parameters, on, after):
    """States from unit central infusion rate for 'on', then washout 'after'."""
    _, k12, _, alpha, beta, gap, a, b = coefficients(log_parameters, on.ndim-1)
    ia = on*exprel_negative(alpha*on)*torch.exp(-alpha*after)
    ib = on*exprel_negative(beta*on)*torch.exp(-beta*after)
    return torch.stack([torch.zeros_like(on), a*ia+b*ib, (k12/gap*(ib-ia)).clamp_min(0)], -1)


def advance(log_parameters, state, start, end, events, include_start=False):
    """Propagate state[B,3] from start[B] to end[B,T].

    Events[B,E,4]: start, amount, route(0 oral/1 bolus/2 infusion), duration.
    Query order and repeated query times have no effect on another query.
    """
    dt = (end-start[:, None]).clamp_min(0)
    evolved = (transition(log_parameters, dt) @ state[:, None, :, None]).squeeze(-1)
    event_t, amount, route, duration = events.unbind(-1)
    include = torch.as_tensor(include_start, device=start.device, dtype=torch.bool).expand_as(start)
    eligible = ((event_t > start[:, None]) | ((event_t == start[:, None]) & include[:, None])) & (amount > 0)
    elapsed = (end[:, :, None]-event_t[:, None, :]).clamp_min(0)
    before_end = event_t[:, None, :] <= end[:, :, None]
    effect = transition(log_parameters, elapsed)
    impulse = torch.where((route == 0)[:, None, :, None], effect[..., :, 0], effect[..., :, 1])
    impulse_mask = eligible[:, None, :] & before_end & (route != 2)[:, None, :]
    bolus = (impulse * (amount[:, None, :]*impulse_mask)[..., None]).sum(2)
    left = torch.maximum(start[:, None], event_t)
    right = torch.minimum(end[:, :, None], (event_t+duration)[:, None, :])
    on = (right-left[:, None, :]).clamp_min(0)
    after = (end[:, :, None]-(event_t+duration)[:, None, :]).clamp_min(0)
    rate = amount/duration.clamp_min(1e-12)*(route == 2)
    infused = (infusion_response(log_parameters, on, after)*rate[:, None, :, None]).sum(2)
    return evolved+bolus+infused


def concentration(log_parameters, states):
    vc = log_parameters[:, 2].exp()
    while vc.ndim < states.ndim-1:
        vc = vc.unsqueeze(-1)
    return torch.log((states[..., 1]/vc).clamp_min(0)+1e-5)
