"""Support-only amplitude and time-scale calibration of a frozen PK prior.

The two directions have an exact meaning in the linear dose-driven ODE:
  time:  multiply ka, CL and Q by exp(a), keeping both volumes unchanged;
  scale: divide CL, Vc, Q and Vp by exp(b), keeping all rates unchanged.
Thus the second direction multiplies concentration by exp(b) before flooring.
No query concentrations, query times or drug labels enter the conditioning solve.
This is a predictive calibration, not identification of physiological parameters.
"""
import math
import torch
from .relational import floor_curve
from .state_ode import advance


DIRECTIONS = ((1., 1., 0., 1., 0.), (0., -1., -1., -1., -1.))
CONFIGS = [dict(name='legacy', kind='legacy')]+[
    dict(name=f'time{t:g}_scale{s:g}', kind='calibrated', time_sd=t, scale_sd=s)
    for t in [0., .3, 1.] for s in [1., 3.]
]


def shift_parameters(prior, offsets):
    return prior+offsets @ prior.new_tensor(DIRECTIONS)


def raw_curve(parameters, events, times):
    start = torch.where(events[..., 1] > 0, events[..., 0], torch.inf).min(1).values
    start = torch.minimum(start, times.min(1).values)
    states = advance(parameters, parameters.new_zeros((len(parameters), 3)), start, times, events, True)
    return states[..., 1]/parameters[:, 2, None].exp()


def _single_curve(offset, prior, events, times):
    return floor_curve(shift_parameters(prior[None], offset[None]), events[None], times[None])[0]


_jacobian = torch.vmap(torch.func.jacfwd(_single_curve, argnums=0))


def pool_support(times, labels, mask):
    """One residual per distinct timestamp; masked entries may contain NaNs."""
    safe_t = torch.where(mask, times, 0.)
    safe_y = torch.where(mask, labels, 0.)
    same = (safe_t[:, :, None] == safe_t[:, None, :]) & mask[:, :, None] & mask[:, None, :]
    n = same.sum(-1).clamp_min(1)
    y = (same*safe_y[:, None, :]).sum(-1)/n
    previous = torch.tril(torch.ones_like(same, dtype=torch.bool), diagonal=-1)
    first = mask & ~(same & previous).any(-1)
    return safe_t, y, first


def calibrate(prior, events, support_t, support_y, support_mask, *, time_sd, scale_sd,
              noise=.3, iterations=6):
    """Two-dimensional damped MAP solve, using observed history only.

    Six deterministic Gauss--Newton steps with support-objective backtracking.
    In zero-information directions the prior is retained. The log-concentration
    floor is inside the objective and forecast. Offset limits only bound the
    numerical search (100-fold time / exp(12)-fold scale).
    """
    if time_sd < 0 or scale_sd <= 0 or noise <= 0 or iterations < 0:
        raise ValueError('Invalid calibration configuration')
    if support_t.shape != support_y.shape or support_mask.shape != support_t.shape:
        raise ValueError('Support tensors must be aligned')
    if not torch.isfinite(support_t[support_mask]).all() or not torch.isfinite(support_y[support_mask]).all():
        raise ValueError('Observed support must be finite')
    if support_t.shape[1] == 0:
        return dict(parameters=prior.clone(), offsets=prior.new_zeros((len(prior), 2)),
                    objective=prior.new_zeros(len(prior)), objective_trace=prior.new_zeros((len(prior), 1)))
    st, sy, sm = pool_support(support_t, support_y, support_mask)
    scales = prior.new_tensor([time_sd, scale_sd])
    offsets = prior.new_zeros((len(prior), 2))
    eye = torch.eye(2, dtype=prior.dtype, device=prior.device).expand(len(prior), -1, -1)

    def objective(white):
        pred = floor_curve(shift_parameters(prior, white*scales), events, st)
        value = .5*torch.where(sm, (pred-sy).square()/noise**2, 0.).sum(1)+.5*white.square().sum(1)
        return torch.where(torch.isfinite(value), value, torch.inf)

    # Exact scale identity supplies an initializer even when an incorrect prior
    # puts positive concentrations below the floor (where the local J is zero).
    raw = raw_curve(prior, events, st)
    informative = sm & (raw > 0)
    amplitude = torch.where(informative, sy-raw.clamp_min(1e-30).log(), 0.).sum(1)
    amplitude = amplitude/(informative.sum(1)+noise**2/scale_sd**2)
    initial = torch.stack([amplitude*0, amplitude.clamp(-12, 12)/scale_sd], -1)
    initial_value, zero_value = objective(initial), objective(offsets)
    white = torch.where((initial_value < zero_value)[:, None], initial, offsets)
    values = [torch.minimum(initial_value, zero_value)]
    for _ in range(iterations):
        offsets = white*scales
        pred = floor_curve(shift_parameters(prior, offsets), events, st)
        j = _jacobian(offsets, prior, events, st)*scales/noise
        j = torch.where(sm[..., None], j, 0.)
        residual = torch.where(sm, (sy-pred)/noise, 0.)
        lhs = eye+j.transpose(1, 2) @ j
        rhs = (j.transpose(1, 2) @ residual[..., None]).squeeze(-1)-white
        step = torch.linalg.solve(lhs, rhs[..., None]).squeeze(-1)
        proposals, scores = [], []
        for alpha in [0., 1., .5, .25, .125]:
            candidate = white+alpha*step
            limits = prior.new_tensor([math.log(100)/max(time_sd, 1e-30), 12/scale_sd])
            candidate = torch.maximum(torch.minimum(candidate, limits), -limits)
            proposals.append(candidate)
            scores.append(objective(candidate))
        scores = torch.stack(scores, 1)
        chosen = scores.argmin(1)
        white = torch.stack(proposals, 1)[torch.arange(len(prior), device=prior.device), chosen]
        values.append(scores[torch.arange(len(prior), device=prior.device), chosen])
    offsets = white*scales
    return dict(parameters=shift_parameters(prior, offsets), offsets=offsets,
                objective=values[-1], objective_trace=torch.stack(values, 1))


def predict(model, inputs, config):
    """Adapter retaining the molecular prior and the original forecasting API."""
    if config['kind'] == 'legacy':
        return model(*inputs, variance=False)
    z, events, st, sy, sm, qt, ids = inputs
    mu = model.prior(z, ids)[0]
    state = calibrate(mu, events, st, sy, sm, time_sd=config['time_sd'], scale_sd=config['scale_sd'])
    return dict(mean=floor_curve(state['parameters'], events, qt), state=state)
