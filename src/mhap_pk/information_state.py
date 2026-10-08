"""Fixed molecular prior, support information update, then state-only replay.

This is a cutoff-conditioned replay: all provided support is used to estimate
parameters before replaying that support. Intermediate replay states are not
claimed to be online forecasts. Query times/labels never enter conditioning.
"""
import torch
from .relational_model import RelationalPK
from .relational_information import FloorInformationBase
from .relational import floor_curve, floor_concentration
from .state_ode import advance


class InformationStatePK(RelationalPK):
    def __init__(self, information_update=True, state_update=True, state_correction='amount'):
        super().__init__(parameter_update=False, state_update=state_update,
                         prior_only=not state_update)
        self.information_update = information_update
        self.state_correction = state_correction
        self.kind = 'v4'
        self.register_buffer('noise', torch.tensor(.3, dtype=torch.float64))
        # Existing one-step information convention; fixed before evaluation.
        self.prior_std = .6
        if state_update:
            for p in self.parameter_head.parameters():
                p.requires_grad_(False)

    def distribution(self, z, drug_indices=None):
        mu = self.prior(z, drug_indices)[0]
        return mu, torch.full_like(mu, self.prior_std)

    def prepare(self, z, events, st, sy, sm, ids=None):
        with torch.no_grad():
            mu, context, chemical = self.prior(z, ids)
            theta = FloorInformationBase.condition(self, z, events, st, sy, sm, ids)['mean'] if self.information_update and self.prior_std > 0 else mu
        return theta.detach(), context.detach(), chemical.detach()

    def forward(self, z, events, st, sy, sm, qt, ids=None, variance=False, prepared=None):
        cutoff = torch.where(sm, st, -torch.inf).max(1).values
        if not torch.isfinite(qt).all() or (qt <= cutoff[:, None]).any():
            raise ValueError('Query must be finite and strictly after support')
        values = self.prepare(z, events, st, sy, sm, ids) if prepared is None else prepared
        if not self.state_update:
            return dict(mean=floor_curve(values[0], events, qt))
        conditioned = super().condition(z, events, st, sy, sm, ids, prior_values=values,
                                        state_correction=self.state_correction)
        states = advance(conditioned['parameters'], conditioned['state'], conditioned['time'], qt,
                         events, conditioned['include_start'])
        return dict(mean=floor_concentration(conditioned['parameters'], states), state=conditioned)
