"""Multi-start nonlinear MAP updates around the learned molecular PK prior.

Unlike a shared time dilation, independent ka/CL/Vc/Q/Vp changes can revise the
relative absorption, distribution and elimination rates. This is an inference
extension of the frozen information prior, not end-to-end retraining.
"""
import torch
from .relational import floor_curve
from .relational_information import batch_jacobian
from .transfer_calibration import raw_curve, pool_support


CONFIGS = [dict(name='legacy', kind='legacy')]+[
    dict(name=f'iterated_sd{sd:g}', kind='iterated', prior_sd=sd)
    for sd in [.6, 1.5, 3.]
]


def condition(prior, events, support_t, support_y, support_mask, *, prior_sd=.6, noise=.3, iterations=8):
    """Select two support-only optimized starts under the SAME MAP objective.

    Eight Gauss--Newton steps; the current iterate is always a candidate. The
    second start uses the exact concentration scaling direction, avoiding zero
    derivatives when an incorrect positive curve falls below the log floor.
    No reference/held-out drug, future labels or query times enter the solve.
    """
    if prior_sd <= 0 or noise <= 0 or iterations < 0:
        raise ValueError('Invalid information update configuration')
    if support_t.shape != support_y.shape or support_t.shape != support_mask.shape:
        raise ValueError('Support must align')
    if not torch.isfinite(support_t[support_mask]).all() or not torch.isfinite(support_y[support_mask]).all():
        raise ValueError('Observed support must be finite')
    n = len(prior)
    if support_t.shape[1] == 0:
        return dict(parameters=prior.clone(), offsets=torch.zeros_like(prior),
                    objective_trace=prior.new_zeros((n, 1)), chosen_start=torch.zeros(n, dtype=torch.long))
    st, sy, sm = pool_support(support_t, support_y, support_mask)
    raw = raw_curve(prior, events, st)
    informative = sm & (raw > 0)
    amplitude = torch.where(informative, sy-raw.clamp_min(1e-30).log(), 0.).sum(1)
    # Four parameters share the exact scaling shift, hence four prior penalties.
    amplitude = amplitude/(informative.sum(1)+4*noise**2/prior_sd**2)
    direction = prior.new_tensor([0., -1., -1., -1., -1.])
    second = amplitude.clamp(-12, 12)[:, None]*direction[None]/prior_sd
    white = torch.cat([torch.zeros_like(prior), second])
    origin, ev, times, labels, mask = [torch.cat([v, v]) for v in [prior, events, st, sy, sm]]
    eye = torch.eye(5, dtype=prior.dtype, device=prior.device).expand(len(origin), -1, -1)

    def objective(w):
        pred = floor_curve(origin+prior_sd*w, ev, times)
        result = .5*torch.where(mask, (pred-labels).square()/noise**2, 0.).sum(1)+.5*w.square().sum(1)
        return torch.where(torch.isfinite(result), result, torch.inf)

    trace = [objective(white)]
    for _ in range(iterations):
        theta = origin+prior_sd*white
        base = floor_curve(theta, ev, times)
        j = batch_jacobian(theta, ev, times)*prior_sd/noise
        j = torch.where(mask[..., None], j, 0.)
        residual = torch.where(mask, (labels-base)/noise, 0.)
        matrix = eye+j.transpose(1, 2) @ j
        rhs = (j.transpose(1, 2) @ residual[..., None]).squeeze(-1)-white
        step = torch.linalg.solve(matrix, rhs[..., None]).squeeze(-1)
        # Numerical trust step; the MAP prior still supplies statistical shrinkage.
        factor = (2./(step*prior_sd).abs().max(1).values.clamp_min(2.)).clamp_max(1.)
        step = step*factor[:, None]
        candidates, values = [], []
        for size in [0., 1., .5, .25, .125]:
            candidate = white+size*step
            if size:
                theta = (origin+prior_sd*candidate).clamp(-16., 20.)
                candidate = (theta-origin)/prior_sd
            candidates.append(candidate); values.append(objective(candidate))
        values = torch.stack(values, 1)
        chosen = values.argmin(1)
        rows = torch.arange(len(origin), device=prior.device)
        white = torch.stack(candidates, 1)[rows, chosen]
        trace.append(values[rows, chosen])
    values = trace[-1].reshape(2, n).T
    starts = values.argmin(1)
    rows = torch.arange(n, device=prior.device)+starts*n
    offsets = white[rows]*prior_sd
    return dict(parameters=prior+offsets, offsets=offsets,
                objective_trace=torch.stack(trace, 1)[rows], chosen_start=starts,
                final_start_objectives=values)


def predict(model, inputs, config):
    if config['kind'] == 'legacy': return model(*inputs, variance=False)
    z, events, st, sy, sm, qt, ids = inputs
    prior = model.prior(z, ids)[0]
    state = condition(prior, events, st, sy, sm, prior_sd=config['prior_sd'])
    return dict(mean=floor_curve(state['parameters'], events, qt), state=state)
