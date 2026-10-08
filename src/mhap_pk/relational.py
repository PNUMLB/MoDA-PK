"""Fitting-only empirical PK relations and a consistent observation link.

Pair labels use fitting query targets, never validation/test curves. Dose
normalization assumes linear exposure in dose. Only single administration with
the identical route and infusion duration is compared, inside measured support.
"""
import numpy as np
import torch
import torch.nn.functional as F
from .state_ode import advance


def floor_concentration(log_parameters, states):
    vc = log_parameters[:, 2].exp()
    while vc.ndim < states.ndim-1:
        vc = vc.unsqueeze(-1)
    return (states[..., 1]/vc).clamp_min(1e-5).log()


def floor_curve(parameters, events, times):
    start = torch.where(events[..., 1] > 0, events[..., 0], torch.inf).min(1).values
    start = torch.minimum(start, times.min(1).values)
    states = advance(parameters, parameters.new_zeros((len(parameters), 3)), start,
                     times, events, True)
    return floor_concentration(parameters, states)


def build_relations(profiles, meta, fitting, vectors, ids):
    """Deterministic detached labels from fitting molecules/participants only."""
    drugs = sorted(meta.iloc[fitting].drug_id.unique())
    first = [next(int(i) for i in fitting if profiles[i]['drug_id'] == drug) for drug in drugs]
    z = vectors[first].detach().clone()
    n = len(drugs)
    # Squared Euclidean distance of fitting-standardized frozen UniMol vectors.
    chemical = torch.cdist(z, z).square()/z.shape[1]
    pk = np.zeros((n, n)); valid = np.zeros((n, n), dtype=bool)
    counts = np.zeros((n, n), dtype=int)
    curves = {d: [] for d in drugs}
    for i in fitting:
        p = profiles[int(i)]
        if len(p['events']) != 1:
            continue
        event = p['events'][0]
        q = p['query'].groupby('time').target_log_dv.mean()
        t = q.index.to_numpy()-event[0]
        keep = t > 0
        if keep.sum() < 3:
            continue
        curves[p['drug_id']].append((float(event[2]), round(float(event[3]), 8),
                                    t[keep], q.to_numpy()[keep]-np.log(event[1]),
                                    p['participant_id']))
    for i in range(n):
        for j in range(i+1, n):
            differences = {}
            for a in curves[drugs[i]]:
                for b in curves[drugs[j]]:
                    if a[:2] != b[:2]:
                        continue
                    lo, hi = max(a[2][0], b[2][0]), min(a[2][-1], b[2][-1])
                    if hi < 2*lo:
                        continue
                    grid = np.geomspace(lo, hi, 12)
                    ya = np.interp(np.log(grid), np.log(a[2]), a[3])
                    yb = np.interp(np.log(grid), np.log(b[2]), b[3])
                    # Whole curve distance retains both level and time shape.
                    value = float(np.mean((ya-yb)**2))
                    differences.setdefault((a[4], b[4]), []).append(value)
            if differences:
                pk[i, j] = pk[j, i] = np.median([np.mean(v) for v in differences.values()])
                counts[i, j] = counts[j, i] = len(differences)
                valid[i, j] = valid[j, i] = True
    chemistry_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return dict(drugs=drugs, z=z, ids=ids[first].detach().clone(),
                chemical_distance=chemical, chemical_mask=chemistry_mask,
                pk_distance=torch.as_tensor(pk, dtype=z.dtype, device=z.device),
                pk_mask=torch.as_tensor(valid, device=z.device), pair_counts=counts,
                fitting_indices=[int(i) for i in fitting],
                contributing_query_ids=sorted({str(row.observation_id) for i in fitting
                    for row in profiles[int(i)]['query'].itertuples()}))


def relation_loss(context, distances, mask):
    """Soft neighborhood KL; anchors with <2 comparisons give zero gradient."""
    good = mask.sum(1) >= 2
    if not good.any():
        return context.sum()*0
    target_logits = torch.zeros_like(distances)
    for i in range(len(mask)):
        if good[i]:
            scale = distances[i, mask[i]].median().clamp_min(.01)
            target_logits[i] = -distances[i]/scale
    target_logits = target_logits[good].masked_fill(~mask[good], -torch.inf)
    targets = target_logits.softmax(-1).detach()
    unit = F.normalize(context, dim=-1)
    logits = (unit @ unit.T)/.2
    logprob = logits[good].masked_fill(~mask[good], -torch.inf).log_softmax(-1)
    # Zero out absent pairs before multiplication to avoid 0 * -inf.
    logprob = torch.where(mask[good], logprob, 0.)
    return (targets*(targets.clamp_min(1e-30).log()-logprob)).sum(1).mean()


def auxiliary_loss(prior, relations, pk_weight, chemical_weight):
    _, context, _ = prior(relations['z'], relations['ids'])
    pk = relation_loss(context, relations['pk_distance'], relations['pk_mask'])
    chem = relation_loss(context, relations['chemical_distance'], relations['chemical_mask'])
    return pk_weight*pk+chemical_weight*chem, pk.detach(), chem.detach()
