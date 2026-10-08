"""Common-origin history: t4; t2,t4; t1,t2,t3,t4, raw replicates retained."""
from dataclasses import replace
import torch


def initial_observation_inputs(batch, k):
    """Evaluate 0..4 initial time blocks without changing training KS or queries.

    The last stored support slice contains all four eligible time blocks,
    including same-time replicates. Masked concentrations are cleared before
    returning model inputs; future query labels are never returned.
    """
    if not isinstance(k, int) or not 0 <= k <= 4:
        raise ValueError('k must be an integer between zero and four')
    full_t = batch.support_t[:, 3]
    full_y = batch.support_y[:, 3]
    full_mask = batch.support_mask[:, 3]
    mask = torch.zeros_like(full_mask)
    for row in range(len(batch.indices)):
        times = torch.unique(full_t[row, full_mask[row]], sorted=True)
        if len(times) != 4:
            raise ValueError('Expected four distinct eligible observation times')
        if not (batch.query_t[row, batch.query_mask[row]] > times[-1]).all():
            raise ValueError('Evaluation queries must be strictly after t4')
        mask[row] = full_mask[row] & torch.isin(full_t[row], times[:k])
    return (batch.z, batch.events, torch.where(mask, full_t, 0.),
            torch.where(mask, full_y, 0.), mask, batch.query_t, batch.ids)

def common_origin(batch):
    t=batch.support_t.clone();y=batch.support_y.clone();mask=batch.support_mask.clone()
    for row in range(len(batch.indices)):
        full_t=batch.support_t[row,3];full_y=batch.support_y[row,3];full_mask=batch.support_mask[row,3]
        times=torch.unique(full_t[full_mask],sorted=True)
        assert len(times)==4
        for ki,selected in enumerate([[],[3],[1,3],[0,1,2,3]]):
            t[row,ki]=full_t;y[row,ki]=full_y
            mask[row,ki]=full_mask & torch.isin(full_t,times[selected])
            if selected:assert full_t[mask[row,ki]].max()==times[-1]
        assert (batch.query_t[row,batch.query_mask[row]]>times[-1]).all()
    return replace(batch,support_t=t,support_y=y,support_mask=mask)
