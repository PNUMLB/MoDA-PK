"""Retrained controls for the molecular-plus-identity MoDE-PK configuration."""
import torch
from .molecular_identity import MolecularIdentityPrior, build_joint_input
from ..adapters.encoder_controls import HistoryCell

class BranchControlPrior(MolecularIdentityPrior):
    def forward(self,z,drug_indices=None):
        if drug_indices is None or ((drug_indices < -1)|(drug_indices>=self.ids.num_embeddings)).any():
            raise ValueError('Training identity or -1 required')
        identity=torch.where((drug_indices>=0)[:,None],self.ids(drug_indices.clamp_min(0)),self.ids.weight.mean(0)[None])
        if self.drop_molecule:z=torch.zeros_like(z)
        if self.drop_identity:identity=torch.zeros_like(identity)
        h=self.project(torch.cat((z,identity),dim=-1));d=self.residual(h)
        return self.base[None]+d,h,d

def build_control(variant,n,pop):
    m=build_joint_input('molecular',n,pop)
    if variant in ['without_updates','parameter_only','state_only']:
        m.parameter_update=variant=='parameter_only'
        m.state_update=variant=='state_only'
    if variant in ['without_molecule','without_identity']:
        rng=torch.get_rng_state();prior=BranchControlPrior(n).double();torch.set_rng_state(rng)
        prior.load_state_dict(m.prior.state_dict());prior.drop_molecule=variant=='without_molecule';prior.drop_identity=variant=='without_identity';m.prior=prior
    if variant=='memoryless':m.gru=HistoryCell('last_mlp').double()
    return m
