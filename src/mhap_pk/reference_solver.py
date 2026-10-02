"""Molecular two-compartment DCM adaptation, not native covariate reproduction.
Parameter order ka,CL,V1,Q,V2. IV and oral share disposition parameters.
Concentration-state equations correspond to pinned two_comp!/two_comp_abs!.
"""
import torch
from torch import nn
class DCMHead(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(512,64),nn.SiLU(),nn.Linear(64,16),nn.SiLU(),nn.Linear(16,5),nn.Softplus())
 def forward(self,x):return self.net(x)
def event_curves(params,events,times):
 if params.shape!=(5,) or not torch.isfinite(params).all() or (params<=0).any():raise ValueError('Expected positive ka CL V1 Q V2')
 ka,cl,v1,q,v2=params;z=params.new_tensor(0.);t=params.new_tensor([float(x) for x in times])
 if not len(t) or not torch.isfinite(t).all():raise ValueError('Invalid queries')
 k=torch.stack([torch.stack([-(cl+q)/v1,q/v2,ka/v1]),torch.stack([q/v1,-q/v2,z]),torch.stack([z,z,-ka])]);out=torch.zeros_like(t)
 for e in events:
  elapsed=(t-e.time).clamp_min(0);s0=params.new_zeros(3)
  if e.route=='oral':s0[2]=e.amount
  elif e.route=='iv_bolus':s0[0]=e.amount/v1
  elif e.route!='iv_infusion':raise ValueError('Unsupported route')
  if e.route=='iv_infusion':
   force=torch.stack([params.new_tensor(e.amount/e.duration)/v1,z,z]);aug=torch.cat([torch.cat([k,force[:,None]],1),params.new_zeros(1,4)],0);initial=params.new_zeros(4);initial[-1]=1
   delivered=(torch.matrix_exp(aug[None]*elapsed.clamp_max(e.duration)[:,None,None])@initial)[:,:3]
   states=(torch.matrix_exp(k[None]*(elapsed-e.duration).clamp_min(0)[:,None,None])@delivered[:,:,None]).squeeze(-1)
  else:states=torch.matrix_exp(k[None]*elapsed[:,None,None])@s0
  out+=torch.where(t>=e.time,states[:,0],torch.zeros_like(t))
 return out
