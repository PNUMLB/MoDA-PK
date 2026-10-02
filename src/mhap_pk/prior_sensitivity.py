"""Prespecified follow-up removing only molecular-deviation saturation.

The main forecast_v2 experiment remains frozen. This is a separate diagnostic
motivated by saturation in fitting-drug priors, not a replacement of its scores.
"""
import torch
from .sequential import MolecularPrior, SequentialPK, MatchedInformationPK


class UnsaturatedMolecularPrior(MolecularPrior):
    def forward(self, z, drug_indices=None):
        if self.ids is not None:
            known = drug_indices >= 0
            z = torch.where(known[:, None], self.ids(drug_indices.clamp_min(0)), self.ids.weight.mean(0)[None])
        context = self.project(z)
        deviation = self.residual(context)
        return self.base[None]+deviation, context, deviation


def replace_prior(model, id_count, hidden):
    # Exact same initialization, including GRU and output heads, as its bounded
    # counterpart. Only the molecular forward map changes.
    state = model.prior.state_dict()
    model.prior = UnsaturatedMolecularPrior(id_count=id_count, hidden=hidden)
    model.prior.load_state_dict(state)


class UnsaturatedSequentialPK(SequentialPK):
    def __init__(self, id_count=0, hidden=32, **kwargs):
        super().__init__(id_count=id_count, hidden=hidden, **kwargs)
        replace_prior(self, id_count, hidden)


class UnsaturatedInformationPK(MatchedInformationPK):
    def __init__(self, id_count=0, hidden=32):
        super().__init__(id_count=id_count, hidden=hidden)
        replace_prior(self, id_count, hidden)
