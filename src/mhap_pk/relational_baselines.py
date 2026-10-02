"""Matched forecasting baselines; official-model adapters are labelled as such."""
import torch
from torch import nn
from .relational_model import CalibratedPrior
from .relational import floor_curve


def event_features(events, times):
    dt = times[:, :, None]-events[:, None, :, 0]
    past = (dt >= 0) & (events[:, None, :, 1] > 0)
    infusion = events[:, None, :, 2] == 2
    duration = events[:, None, :, 3].clamp_min(1e-12)
    delivered = events[:, None, :, 1]*torch.where(infusion, (dt/duration).clamp(0, 1), 1.)*past
    rate = (events[:, None, :, 1]/duration*infusion*past*(dt < duration)).sum(-1)
    elapsed = torch.where(past.any(-1), torch.where(past, dt, torch.inf).min(-1).values, 0.)
    routes = [(past & (events[:, None, :, 2] == r)).any(-1) for r in range(3)]
    return torch.stack([delivered.sum(-1).log1p(), rate.log1p(), elapsed.log1p(), *routes], -1)


class HistoryGRU(nn.Module):
    def __init__(self, id_count=0):
        super().__init__()
        self.prior = CalibratedPrior(id_count=id_count)
        self.gru = nn.GRUCell(32+9, 32)
        self.readout = nn.Sequential(nn.Linear(32+32+8, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.readout[-1].weight); nn.init.zeros_(self.readout[-1].bias)

    def forward(self, z, events, st, sy, sm, qt, ids=None, variance=False):
        mu, context, _ = self.prior(z, ids)
        b = len(z); h = z.new_zeros((b, 32))
        first = torch.where(events[..., 1] > 0, events[..., 0], torch.inf).min(1).values
        previous = first.clone(); last_y = z.new_zeros(b)
        any_seen = sm.any(1)
        order = torch.argsort(torch.where(sm, st, torch.inf), stable=True)
        st, sy, sm = [v.gather(1, order) for v in [st, sy, sm]]
        ef = event_features(events, st)
        for i in range(st.shape[1]):
            active = sm[:, i] & ((i == 0) | (st[:, i] != st[:, max(0, i-1)]))
            same = sm & (st == st[:, i, None]); count = same.sum(1).clamp_min(1)
            y = torch.where(same, sy, 0.).sum(1)/count
            features = torch.cat([context, y[:, None], (st[:, i]-previous).clamp_min(0).log1p()[:, None],
                                  count.log1p()[:, None], ef[:, i]], -1)
            proposal = self.gru(features, h)
            h = torch.where(active[:, None], proposal, h)
            previous = torch.where(active, st[:, i], previous)
            last_y = torch.where(active, y, last_y)
        future = event_features(events, qt)
        dt = (qt-previous[:, None]).clamp_min(0)
        features = torch.cat([context[:, None].expand(-1, qt.shape[1], -1),
            h[:, None].expand(-1, qt.shape[1], -1), future, dt.log1p()[..., None],
            (qt-first[:, None]).clamp_min(0).log1p()[..., None]], -1)
        dose = torch.expm1(future[..., 0]).clamp_min(1e-5)
        base = torch.where(any_seen[:, None], last_y[:, None], dose.log()-mu[:, 2, None])
        return dict(mean=base+self.readout(features).squeeze(-1), state={}, regularization=z.new_zeros(b))


class PopulationMAP(nn.Module):
    """Learned shared population parameters, support-only regularized log scale."""
    def __init__(self, id_count=0):
        super().__init__()
        self.prior = CalibratedPrior(id_count=id_count)

    def forward(self, z, events, st, sy, sm, qt, ids=None, variance=False):
        mu, _, _ = self.prior(z, ids)
        prediction = floor_curve(mu, events, qt)
        past = floor_curve(mu, events, st)
        shift = torch.where(sm, sy-past, 0.).sum(1)/(sm.sum(1)+.09)
        return dict(mean=prediction+shift[:, None], state={}, regularization=z.new_zeros(len(z)))
