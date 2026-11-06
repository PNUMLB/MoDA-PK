"""Mamba-1 observer, differentiable non-fused recurrence in the PK solver dtype.

Architecture/initialization follow state-spaces/mamba mamba_simple.py.
Reference: https://github.com/state-spaces/mamba (Apache-2.0).
Unlike the official inference step, states are functional (no in-place copy)
and remain attached to autograd during training. No extra norm/residual block.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class MambaBlock(nn.Module):
    def __init__(self,d_model=32,d_state=16,d_conv=4,expand=2):
        super().__init__()
        self.d_inner=d_model*expand;self.d_state=d_state;self.d_conv=d_conv
        self.dt_rank=math.ceil(d_model/16)
        self.in_proj=nn.Linear(d_model,2*self.d_inner,bias=False)
        self.conv1d=nn.Conv1d(self.d_inner,self.d_inner,d_conv,groups=self.d_inner,padding=d_conv-1)
        self.x_proj=nn.Linear(self.d_inner,self.dt_rank+2*d_state,bias=False)
        self.dt_proj=nn.Linear(self.dt_rank,self.d_inner)
        nn.init.uniform_(self.dt_proj.weight,-self.dt_rank**-.5,self.dt_rank**-.5)
        dt=torch.exp(torch.rand(self.d_inner)*(math.log(.1)-math.log(.001))+math.log(.001)).clamp_min(1e-4)
        with torch.no_grad():self.dt_proj.bias.copy_(dt+torch.log(-torch.expm1(-dt)))
        self.A_log=nn.Parameter(torch.arange(1,d_state+1,dtype=torch.float32).log().repeat(self.d_inner,1))
        self.D=nn.Parameter(torch.ones(self.d_inner))
        self.out_proj=nn.Linear(self.d_inner,d_model,bias=False)

    def step(self,token,conv,ssm):
        x,z=self.in_proj(token).chunk(2,-1)
        conv=torch.cat((conv[:,:,1:],x[:,:,None]),-1)
        x=F.silu((conv*self.conv1d.weight[:,0,:]).sum(-1)+self.conv1d.bias)
        dt,b,c=self.x_proj(x).split([self.dt_rank,self.d_state,self.d_state],-1)
        dt=F.softplus(self.dt_proj(dt))
        ssm=torch.exp(-dt[:,:,None]*self.A_log.exp()[None])*ssm+dt[:,:,None]*b[:,None,:]*x[:,:,None]
        y=(ssm*c[:,None,:]).sum(-1)+self.D*x
        return self.out_proj(y*F.silu(z)),conv,ssm

class MambaObserver(nn.Module):
    def __init__(self,inputs=43,hidden=32):
        super().__init__();self.embed=nn.Linear(inputs,hidden);self.block=MambaBlock(hidden)
    def reset_history(self):self.conv=None;self.ssm=None
    def set_active(self,active):self.active=active
    def forward(self,x,h):
        if self.conv is None:
            self.conv=x.new_zeros((len(x),self.block.d_inner,self.block.d_conv))
            self.ssm=x.new_zeros((len(x),self.block.d_inner,self.block.d_state))
        y,conv,ssm=self.block.step(self.embed(x),self.conv,self.ssm)
        mask=self.active[:,None,None]
        self.conv=torch.where(mask,conv,self.conv);self.ssm=torch.where(mask,ssm,self.ssm)
        return torch.where(self.active[:,None],y,h)

def build_mamba(mode,n,pop):
    from ..priors.molecular_identity import build_joint_input
    model=build_joint_input('molecular',n,pop)
    model.gru=MambaObserver().double()
    return model
