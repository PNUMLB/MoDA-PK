"""Molecular and training-drug identity conditioning with explicit unseen fallback."""
import torch
from torch import nn
from ...relational_model import CalibratedPrior, RelationalPK

class MolecularIdentityPrior(CalibratedPrior):
    def __init__(self, n_drugs, hidden=32):
        if n_drugs < 1:
            raise ValueError('At least one training drug is required')
        super().__init__(hidden=hidden)
        self.ids = nn.Embedding(n_drugs, 512)
        self.project = nn.Sequential(nn.Linear(1024, hidden), nn.SiLU())

    def forward(self, z, drug_indices=None):
        if drug_indices is None or ((drug_indices < -1) | (drug_indices >= self.ids.num_embeddings)).any():
            raise ValueError('Use a training drug index or -1 for unseen drugs')
        identity = torch.where((drug_indices >= 0)[:, None],
                               self.ids(drug_indices.clamp_min(0)),
                               self.ids.weight.mean(0)[None])
        context = self.project(torch.cat((z, identity), dim=-1))
        deviation = self.residual(context)
        return self.base[None] + deviation, context, deviation

def build_joint_input(mode, n_drugs, population):
    model = RelationalPK().double()
    model.prior = MolecularIdentityPrior(n_drugs).double()
    with torch.no_grad():
        model.prior.base.copy_(population)
    return model
