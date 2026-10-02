"""Explicit implementations for the low-data comparator panel.

DeepCt/Uni-PK retain the previously pinned matched-task implementations. Neither
is described as native paper reproduction. DCM is deterministic, not ME-DCM.
"""
import torch
from torch import nn
from ..models.priors.low_data import make_low_data_model
from ..relational_model import CalibratedPrior
from ..relational import floor_curve
from ..literature_adapters import DeepCtAdapter
from ..unipk_cached import CachedUniPKAdapter

HEADS=['no_molecule','drug_id','molecular','molecular_offset','scale_rate']
class DeterministicDCM(nn.Module):
    def __init__(self):
        super().__init__();self.prior=CalibratedPrior()
    def forward(self,z,events,st,sy,sm,qt,ids=None):
        mu,_,_=self.prior(z,ids);past=floor_curve(mu,events,st)
        shift=torch.where(sm,sy-past,0.).sum(1)/(sm.sum(1)+.09)
        return dict(mean=floor_curve(mu,events,qt)+shift[:,None])

def build(mode,n,pop):
    if mode in HEADS:return make_low_data_model(mode,n,pop)
    m={'dcm':DeterministicDCM,'deepct':DeepCtAdapter,'unipk':CachedUniPKAdapter}[mode]().double()
    with torch.no_grad():
        m.prior.base.copy_(torch.as_tensor(pop,dtype=torch.double))
        if hasattr(m,'initialize'):m.initialize(torch.as_tensor(pop,dtype=torch.double))
    return m
