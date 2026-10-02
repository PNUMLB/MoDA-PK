"""Bounded molecular conditioning of the existing 11 observer features."""
import torch
from torch import nn
from ...relational_model import RelationalPK

class MolecularGRU(nn.Module):
    def __init__(self,cell,hidden=32,bound=.25):
        super().__init__();self.cell=cell;self.modulation=nn.Linear(hidden,22);self.bound=bound
        nn.init.zeros_(self.modulation.weight);nn.init.zeros_(self.modulation.bias)
    def forward(self,x,h):
        features,context=x[:,:11],x[:,11:]
        gamma,beta=self.modulation(context).chunk(2,-1)
        adjusted=features*(1+self.bound*gamma.tanh())+self.bound*beta.tanh()
        return self.cell(torch.cat([adjusted,context],-1),h)

class MoleculeConditionedPK(RelationalPK):
    def __init__(self):
        super().__init__();self.gru=MolecularGRU(self.gru,self.hidden)

def make_model(mode):
    if mode in ['molecule_film','film_within_drug']:return MoleculeConditionedPK()
    if mode in ['native','within_drug']:return RelationalPK()
    raise ValueError(mode)
