"""Low-data head controls; all use the same explicit PK dynamics and observer."""
import torch
from torch import nn
from ...relational_model import CalibratedPrior,RelationalPK

class MolecularDrugOffset(CalibratedPrior):
    def __init__(self,n):
        super().__init__();self.offset=nn.Embedding(n,5);nn.init.zeros_(self.offset.weight)
    def forward(self,z,drug_indices=None):
        mu,context,deviation=super().forward(z,drug_indices)
        known=drug_indices>=0;offset=2*self.offset(drug_indices.clamp_min(0)).tanh()*known[:,None]
        return mu+offset,context,deviation+offset

class ScaleRatePrior(nn.Module):
    """Predict log Vc and rates ka, CL/Vc, Q/Vc, Q/Vp; no extra amplitude freedom."""
    def __init__(self):
        super().__init__();self.project=nn.Sequential(nn.Linear(512,32),nn.SiLU());self.rate=nn.Linear(32,4);self.scale=nn.Linear(32,1);self.base=nn.Parameter(torch.zeros(5))
        for m in [self.rate,self.scale]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,z,drug_indices=None):
        h=self.project(z);offset=torch.cat([self.rate(h),self.scale(h)],-1);x=self.base[None]+offset
        ka,k10,k12,k21,vc=x.unbind(-1);mu=torch.stack([ka,k10+vc,vc,k12+vc,k12+vc-k21],-1)
        b=self.base;baseline=torch.stack([b[0],b[1]+b[4],b[4],b[2]+b[4],b[2]+b[4]-b[3]])
        return mu,h,mu-baseline

def make_low_data_model(mode,n_drugs,population):
    m=RelationalPK(id_count=n_drugs if mode=='drug_id' else 0).double()
    original=m.prior
    if mode=='molecular_offset':m.prior=MolecularDrugOffset(n_drugs).double()
    elif mode=='scale_rate':m.prior=ScaleRatePrior().double()
    elif mode not in ['molecular','no_molecule','drug_id']:raise ValueError(mode)
    if mode in ['molecular_offset','scale_rate']:m.prior.project.load_state_dict(original.project.state_dict())
    t=torch.as_tensor(population,dtype=torch.double)
    with torch.no_grad():
        if mode=='scale_rate':m.prior.base.copy_(torch.stack([t[0],t[1]-t[2],t[3]-t[2],t[3]-t[4],t[2]]))
        else:m.prior.base.copy_(t)
    return m
