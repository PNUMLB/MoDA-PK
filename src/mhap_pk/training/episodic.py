"""First-order leave-source-out meta training for the unchanged PK model.

Meta-B labels supervise training only. This is not an independent holdout:
the shared initialization is updated across episodes containing every source.
"""
import numpy as np
import torch
from torch.func import functional_call


def source_episodes(meta, fitting):
    fitting = np.asarray(fitting)
    rows = meta.iloc[fitting]
    result = []
    for source in sorted(rows.source_id.unique()):
        b = fitting[rows.source_id.to_numpy() == source]
        a = fitting[rows.source_id.to_numpy() != source]
        if not len(a) or not len(b):
            raise ValueError('At least two fitting sources required')
        ma, mb = meta.iloc[a], meta.iloc[b]
        if set(ma.drug_id) & set(mb.drug_id):
            raise ValueError('A/B share drugs: use connected source-drug grouping')
        keys = lambda x: set(zip(x.source_id, x.participant_id))
        if keys(ma) & keys(mb):
            raise ValueError('A/B share participants')
        result.append((str(source), a, b))
    return result


def forecast_loss(model, batch, ks, parameters=None):
    args = batch.inputs(ks)
    out = model(*args) if parameters is None else functional_call(model, parameters, args, strict=True)
    mse = ((out['mean'] - batch.query_y).square() * batch.weights).sum()
    w = batch.weights.sum(1)
    personal = (out['regularization'] * w).sum()
    deviation = model.prior(batch.z, batch.ids)[2] if parameters is None else out['state']['chemical_offset']
    chemical = (deviation.square().mean(1) * w).sum()
    return mse + .001 * personal + .0001 * chemical, mse


def first_order_step(model, loss_a, step_size, clip=5.):
    """Temporary SGD; detached inner gradient retains identity outer derivative."""
    if step_size < 0 or clip <= 0:
        raise ValueError('Invalid step size/clip')
    parameters = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss_a, tuple(parameters.values()), create_graph=False)
    norm = torch.sqrt(sum(g.detach().square().sum() for g in gradients))
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite inner gradient')
    scale = (clip / (norm + 1e-6)).clamp(max=1.)
    fast = {key: value - step_size * grad.detach() * scale
            for (key, value), grad in zip(parameters.items(), gradients)}
    return fast, norm, step_size * norm * scale
