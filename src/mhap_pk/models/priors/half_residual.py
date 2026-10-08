"""Known-only half residual; unknown predictions use the shared prior exactly."""
import torch
from .low_data import MolecularDrugOffset,make_low_data_model
from ...relational_model import CalibratedPrior

class HalfDrugResidual(MolecularDrugOffset):
    def forward(self,z,drug_indices=None):
        mu,context,deviation=CalibratedPrior.forward(self,z,drug_indices)
        known=drug_indices>=0
        # b_d = 2*tanh(w_d); mu + 0.5*b_d. Unknown has alpha=0.
        delta=self.offset(drug_indices.clamp_min(0)).tanh()*known[:,None]
        return mu+delta,context,deviation+delta

def make_half_residual(n,population,parent_state=None):
    model=make_low_data_model('molecular_offset',n,population)
    # Identical parameters and RNG consumption; only the forward equation changes.
    with torch.random.fork_rng(devices=[]):
        prior=HalfDrugResidual(n).double()
    prior.load_state_dict(model.prior.state_dict());model.prior=prior
    if parent_state is not None:
        missing,unexpected=model.load_state_dict(parent_state,strict=False)
        if missing!=['prior.offset.weight'] or unexpected:
            raise ValueError(f'Unexpected parent state: {missing}, {unexpected}')
        for name,p in model.named_parameters():p.requires_grad_(name=='prior.offset.weight')
    return model
