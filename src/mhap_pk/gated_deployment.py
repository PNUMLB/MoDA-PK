"""Raw-input deployment wrapper for the frozen, cross-fitted v7 gate.

Only molecule vectors/IDs, dose events, observed history and query TIMES enter
forward. Future concentration targets are absent from the inference interface.
"""
import torch
from torch import nn
from .soft_hybrid import FrozenExpertSoftHybrid,mix_dynamics
from .gated_study import ReliabilityGate
from .relational import floor_curve,floor_concentration
from .state_ode import advance


class GatedPK(nn.Module):
    def __init__(self,sequential,information,registry,gate_state,representation='unimol',fusion='dynamics',information_config=None):
        super().__init__()
        self.experts=FrozenExpertSoftHybrid(sequential,information,registry,information_config,fusion)
        # The prototype's unused initialization is not part of this learned gate.
        for p in self.experts.gate.parameters():p.requires_grad_(False)
        self.representation=representation;self.fusion=fusion
        self.gate=ReliabilityGate(gate_state['feature_mean'],gate_state['feature_sd'])
        self.gate.load_state_dict(gate_state)

    def forward(self,z,events,st,sy,sm,qt,ids=None):
        cutoff=torch.where(sm,st,-torch.inf).max(1).values if st.shape[1] else z.new_full((len(z),),-torch.inf)
        if qt.ndim!=2 or qt.shape[0]!=len(z) or not qt.shape[1] or not torch.isfinite(qt).all() or (qt<=cutoff[:,None]).any():
            raise ValueError('Finite future query times strictly after observed support required')
        state=self.experts.condition(z,events,st,sy,sm,ids)
        features=state['gate_features'].clone()
        if self.representation=='id':
            if ids is None:raise ValueError('ID control requires input identity indices')
            features[:,0]=(ids<0).to(z.dtype).log1p()
        elif self.representation=='none':features[:,0]=0.
        alpha=self.gate(features)
        theta,amount=mix_dynamics(state['sequential_parameters'],state['sequential_state'],state['information_parameters'],state['information_state'],alpha)
        if self.fusion=='dynamics':
            propagated=advance(theta,amount,state['time'],qt,events,state['include_start'])
            mean=floor_concentration(theta,propagated)
        else:
            propagated=advance(state['sequential_parameters'],state['sequential_state'],state['time'],qt,events,state['include_start'])
            ys=floor_concentration(state['sequential_parameters'],propagated)
            yi=floor_curve(state['information_parameters'],events,qt)
            mean=(1-alpha[:,None])*ys+alpha[:,None]*yi
        return dict(mean=mean,alpha=alpha,features=features,parameters=theta,state=amount,
            time=state['time'],include_start=state['include_start'],fusion=self.fusion)
