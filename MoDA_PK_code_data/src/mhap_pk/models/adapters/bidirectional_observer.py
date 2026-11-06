"""Bidirectional encoding of the AVAILABLE observation prefix only.

Two independent width-32 directions; concatenate terminal summaries and project
64->32 for unchanged PK heads. BiMamba means this explicit two-direction
Mamba-1 control, not a claimed reproduction of a named BiMamba paper.
Tokens contain sequentially computed PK features; no later observation or
query is replayed into an earlier PK update.
"""
import torch
from torch import nn
from .mamba_observer import MambaBlock

class BidirectionalObserver(nn.Module):
    def __init__(self,kind,inputs=43,hidden=32):
        super().__init__();self.kind=kind;self.hidden=hidden
        if kind=='bimamba':
            self.embed=nn.Linear(inputs,hidden)
            self.directions=nn.ModuleList([MambaBlock(hidden),MambaBlock(hidden)])
        elif kind in ['bigru','bilstm']:
            cls=nn.GRUCell if kind=='bigru' else nn.LSTMCell
            self.directions=nn.ModuleList([cls(inputs,hidden),cls(inputs,hidden)])
        else:raise ValueError(kind)
        self.project=nn.Linear(2*hidden,hidden)
    def reset_history(self):self.tokens=[];self.masks=[]
    def set_active(self,active):self.active=active
    def forward(self,x,previous):
        self.tokens.append(x);self.masks.append(self.active)
        tokens=torch.stack(self.tokens,1);mask=torch.stack(self.masks,1)
        if self.kind=='bimamba':tokens=self.embed(tokens)
        summaries=[]
        for direction,cell in enumerate(self.directions):
            h=x.new_zeros((len(x),self.hidden));c=torch.zeros_like(h)
            if self.kind=='bimamba':
                conv=x.new_zeros((len(x),cell.d_inner,cell.d_conv))
                state=x.new_zeros((len(x),cell.d_inner,cell.d_state))
            indices=range(len(self.tokens)) if direction==0 else reversed(range(len(self.tokens)))
            for i in indices:
                active=mask[:,i,None]
                if self.kind=='bimamba':
                    proposed,newconv,newstate=cell.step(tokens[:,i],conv,state)
                    conv=torch.where(active[:,:,None],newconv,conv)
                    state=torch.where(active[:,:,None],newstate,state)
                elif self.kind=='bilstm':
                    proposed,newc=cell(tokens[:,i],(h,c));c=torch.where(active,newc,c)
                else:proposed=cell(tokens[:,i],h)
                h=torch.where(active,proposed,h)
            summaries.append(h)
        y=self.project(torch.cat(summaries,-1))
        return torch.where(self.active[:,None],y,previous)

def build_bidirectional(kind,n,pop):
    from ..priors.molecular_identity import build_joint_input
    model=build_joint_input('molecular',n,pop)
    model.gru=BidirectionalObserver(kind).double()
    return model
