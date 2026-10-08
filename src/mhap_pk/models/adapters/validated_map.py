"""Support-only bounded MAP controls with per-episode checkpoint selection.

Amplitude is an explicit additive log-output shift (not a PK parameter).
Fixed observation noise and scale are controls, not fitted NLME variances.
"""
import torch
from ...relational import floor_curve
from .joint_estimator import block_support

def fit_map(prior, args, mode='cl_vc', scale=.3, steps=60, noise=.3):
    z,events,t,y,mask,_,*rest=args
    with torch.no_grad():mu=prior(z,rest[0] if rest else None)[0]
    t,y,mask,_,_=block_support(t,y,mask,events)
    active=torch.zeros_like(mu)
    if mode=='cl_vc':active[:,1:3]=1
    elif mode=='full':
        active[:]=1
        cutoff=torch.where(mask,t,-torch.inf).max(1).values
        oral=((events[:,:,1]>0)&(events[:,:,2]==0)&(events[:,:,0]<=cutoff[:,None])).any(1)
        active[:,0]=oral
    elif mode!='amplitude':raise ValueError(mode)
    if scale<=0 or noise<=0:raise ValueError('Positive scales required')
    u=mu.new_zeros((len(mu),1 if mode=='amplitude' else 5),requires_grad=True)
    opt=torch.optim.Adam([u],lr=.03)
    def objective(v):
        pred=floor_curve(mu if mode=='amplitude' else mu+v*active,events,t)
        if mode=='amplitude':pred=pred+v
        return .5*(torch.where(mask,((pred-y)/noise).square(),0.)).sum(1)+.5*(v/scale).square().sum(1)
    with torch.no_grad():best=u.detach().clone();best_obj=objective(best);initial=best_obj.clone();failed=~torch.isfinite(best_obj)
    for _ in range(steps):
        opt.zero_grad();obj=objective(u)
        if not torch.isfinite(obj).all():failed|=~torch.isfinite(obj);break
        obj.sum().backward()
        if not torch.isfinite(u.grad).all():failed|=~torch.isfinite(u.grad).all(1);break
        opt.step()
        with torch.no_grad():
            u.clamp_(-2.,2.);current=objective(u);better=torch.isfinite(current)&(current<best_obj)
            best=torch.where(better[:,None],u.detach(),best);best_obj=torch.where(better,current,best_obj)
    return dict(parameters=mu if mode=='amplitude' else mu+best*active,shift=best if mode=='amplitude' else mu.new_zeros(len(mu),1),initial_objective=initial,objective=best_obj,update=best,failed=failed,active=active)

def predict_map(fit,events,t):return floor_curve(fit['parameters'],events,t)+fit['shift']
