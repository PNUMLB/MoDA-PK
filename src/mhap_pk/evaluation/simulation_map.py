"""Fixed-budget CL/Vc MAP diagnostic; no future-query input to adaptation."""
import torch
from ..relational import floor_curve
from ..baselines.population_vi import pool_support

def adapt(mu,events,st,sy,mask,steps=50,prior_sd=.6,noise_sd=.3):
 sy,mask=pool_support(st,sy,mask);u=torch.zeros(len(mu),2,dtype=mu.dtype,requires_grad=True);best_u=u.detach().clone();best=torch.full((len(mu),),float('inf'),dtype=mu.dtype)
 def obj(v):
  pred=floor_curve(mu+torch.nn.functional.pad(v,(1,2)),events,st)
  return .5*torch.where(mask,((pred-sy)/noise_sd).square(),0.).sum(1)+.5*(v/prior_sd).square().sum(1)
 with torch.enable_grad():
  optimizer=torch.optim.Adam([u],lr=.05)
  for step in range(steps+1 if mask.any() else 1):
   o=obj(u);assert torch.isfinite(o).all();better=o.detach()<best;best=torch.minimum(best,o.detach());best_u=torch.where(better[:,None],u.detach(),best_u)
   if step==steps or not mask.any():break
   optimizer.zero_grad();o.sum().backward();torch.nn.utils.clip_grad_norm_([u],20,error_if_nonfinite=True);optimizer.step()
   with torch.no_grad():u.clamp_(-3,3)
 return mu+torch.nn.functional.pad(best_u,(1,2)),dict(objective=best,prior_objective=obj(torch.zeros_like(u)).detach(),delta=best_u)
