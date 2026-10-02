"""Molecular prior and support-observation-conditioned positive PK parameters.

No query concentration is accepted. Adapt once using support, then evaluate the
same compartment system at arbitrary future times. This is an unbenchmarked
prototype, distinct from the completed bounded concentration-residual model.
"""
from pathlib import Path
import sys

import torch
from torch import nn

from .reference_solver import event_curves


class MolecularHistoryCompartmentModel(nn.Module):
    def __init__(self, embedding_dim=512, hidden=32, offset_bound=1.0):
        super().__init__()
        if offset_bound <= 0:
            raise ValueError('offset_bound must be positive')
        self.embedding_dim = embedding_dim
        self.offset_bound = offset_bound
        self.prior = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.SiLU(), nn.Linear(hidden, 5))
        self.support_encoder = nn.Sequential(nn.Linear(11, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.offset = nn.Sequential(nn.Linear(hidden + 5, hidden), nn.SiLU(), nn.Linear(hidden, 5))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    @staticmethod
    def positive(raw):
        return torch.nn.functional.softplus(raw) + 1e-6

    @staticmethod
    def log_curve(parameters, events, times):
        return torch.log(event_curves(parameters, events, times).clamp_min(0) + 1e-5)

    def parameters_from_history(self, embedding, events, support_times, support_log_c):
        if embedding.shape != (self.embedding_dim,) or not torch.isfinite(embedding).all():
            raise ValueError('Expected one finite frozen molecular vector')
        if support_times.ndim != 1 or support_log_c.shape != support_times.shape:
            raise ValueError('Support times and concentrations must be aligned vectors')
        if not torch.isfinite(support_times).all() or not torch.isfinite(support_log_c).all():
            raise ValueError('Nonfinite support')
        raw = self.prior(embedding)
        if not len(support_times):
            return self.positive(raw), torch.zeros_like(raw)
        base = self.log_curve(self.positive(raw), events, support_times.tolist())
        dose_rows = []
        for time in support_times.tolist():
            past = [event for event in events if event.time <= time]
            # For an ongoing infusion only delivered amount is in the history.
            amounts = [event.amount * min((time - event.time) / event.duration, 1.)
                       if event.route == 'iv_infusion' else event.amount for event in past]
            rate = sum(event.amount / event.duration for event in past
                       if event.route == 'iv_infusion' and time < event.time + event.duration)
            elapsed = time - max(event.time for event in past) if past else 0.
            dose_rows.append([sum(amounts), elapsed, rate, len(past),
                              *[float(any(event.route == route for event in past))
                                for route in ['oral', 'iv_bolus', 'iv_infusion']]])
        dose = embedding.new_tensor(dose_rows)
        dose = torch.cat([torch.log1p(dose[:, :4]), dose[:, 4:]], dim=1)
        features = torch.cat([support_times[:, None] / 24., support_log_c[:, None],
                              base[:, None], (support_log_c - base)[:, None], dose], dim=1)
        pooled = self.support_encoder(features).mean(dim=0)
        delta = self.offset_bound * torch.tanh(self.offset(torch.cat([pooled, raw])))
        return self.positive(raw + delta), delta

    def forward(self, embedding, events, support_times, support_log_c, query_times):
        if query_times.ndim != 1 or not len(query_times) or not torch.isfinite(query_times).all():
            raise ValueError('Expected finite future query times')
        if len(support_times) and (query_times <= support_times.max()).any():
            raise ValueError('Queries must be strictly after all support observations')
        parameters, delta = self.parameters_from_history(embedding, events, support_times, support_log_c)
        prediction = self.log_curve(parameters, events, query_times.tolist())
        return dict(log_concentration=prediction, parameters=parameters, parameter_offset=delta)
