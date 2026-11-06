"""Small causal dose-conditioned controls of the existing shared GRU observer.

No route-specific experts, no test-tuned routing, no change to the PK solver.
Dose features: log1p interval, time since last dose, delivered amount, active
infusion rate, and three historical route flags (mixed regimens allowed).
"""
import torch
from torch import nn
from ...relational_model import RelationalPK
from ..priors.molecular_identity import build_joint_input

DOSE_FEATURES=(3,4,5,6,8,9,10)


class DosingConditionedPK(RelationalPK):
    def observer_features(self, features, context):
        if self.dosing_kind != 'dose_film':
            return features
        raw=self.dose_controller(features[:,DOSE_FEATURES])
        scale,shift=raw.chunk(2,-1)
        return (1+.5*scale.tanh())*features+.5*shift.tanh()

    def observer_updates(self, hidden, features, context):
        if self.dosing_kind != 'dose_gate':
            return super().observer_updates(hidden,features,context)
        control=self.dose_controller(torch.cat([hidden,context,features[:,DOSE_FEATURES]],-1))
        # Neutral at zero control; bounded log-parameter offsets remain +/-2.
        parameter_scale=2*control[:,:1].sigmoid()
        offset=self.personal_bound*torch.tanh(parameter_scale*self.parameter_head(hidden))
        # Modify the existing state-gain logit, rather than multiply two gates.
        gain=(self.gain_head(hidden)+control[:,1:2]).sigmoid().squeeze(-1)
        return offset,gain


def build_dosing(kind,n_drugs,population):
    if kind not in ['dose_film','dose_gate']:raise ValueError(kind)
    # Preserve exact baseline initialization and subsequent random-number state.
    baseline=build_joint_input('molecular',n_drugs,population)
    with torch.random.fork_rng(devices=[]):
        model=DosingConditionedPK().double()
        model.prior=baseline.prior
        model.load_state_dict(baseline.state_dict())
        model.dosing_kind=kind
        model.dose_controller=nn.Sequential(nn.Linear(7 if kind=='dose_film' else 71,16),nn.SiLU(),nn.Linear(16,22 if kind=='dose_film' else 2)).double()
        nn.init.zeros_(model.dose_controller[-1].weight)
        nn.init.zeros_(model.dose_controller[-1].bias)
    return model
