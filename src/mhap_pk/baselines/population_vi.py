"""Drug-specific two-compartment nonlinear mixed-effects model.

Diagonal log-normal random effects on CL and Vc, Gaussian log-concentration
noise, mean-field Gaussian variational population fitting, support-only EBE MAP.
A transparent local statistical implementation; not nlmixr or ME-DCM reproduction.
"""
import torch
from torch import nn
from ..relational import floor_curve

class PopulationVI(nn.Module):
    def __init__(self,initial,n_people):
        super().__init__();initial=torch.as_tensor(initial,dtype=torch.double)
        self.mu=nn.Parameter(initial.clone());self.register_buffer('initial',initial.clone())
        self.log_omega=nn.Parameter(torch.full((2,),-.7,dtype=torch.double));self.log_sigma=nn.Parameter(torch.tensor(-.7,dtype=torch.double))
        self.q_mean=nn.Parameter(torch.zeros(n_people,2,dtype=torch.double));self.q_logsd=nn.Parameter(torch.full((n_people,2),-1.5,dtype=torch.double))
    def theta(self,eta):
        return self.mu[None]+torch.nn.functional.pad(eta,(1,2))
    def objective(self,events,times,labels,mask,person,epsilon):
        # All observations contribute to each person's likelihood; no pseudo-persons.
        sd=self.q_logsd.exp();eta=self.q_mean[None]+sd[None]*epsilon
        count=len(person);draws=len(epsilon);theta=self.theta(eta[:,person].reshape(-1,2))
        y=floor_curve(theta,events.repeat(draws,1,1),times.repeat(draws,1))
        err=y-labels.repeat(draws,1);sigma=self.log_sigma.exp()
        nll=torch.where(mask.repeat(draws,1),.5*(err/sigma).square()+self.log_sigma,0.).sum()/draws
        kl=(self.log_omega[None]-self.q_logsd+.5*((sd.square()+self.q_mean.square())/self.log_omega.exp().square()[None]-1)).sum()
        return (nll+kl)/len(self.q_mean)+1e-4*(self.mu-self.initial).square().mean()
    def constrain(self):
        with torch.no_grad():
            self.mu.clamp_(-12,16);self.log_omega.clamp_(-3, .6931471805599453);self.log_sigma.clamp_(-3,1.0986122886681098);self.q_logsd.clamp_(-5,1);self.q_mean.clamp_(-3,3)

def pool_support(t,y,mask):
    pooled=torch.zeros_like(y);valid=torch.zeros_like(mask)
    for j in range(t.shape[1]):
        same=mask&(t==t[:,j,None]);pooled[:,j]=torch.where(same,y,0.).sum(1)/same.sum(1).clamp_min(1)
        valid[:,j]=mask[:,j]&~((t[:,:j]==t[:,j,None])&mask[:,:j]).any(1)
    return pooled,valid

def map_forecast(model,events,st,sy,sm,qt,steps=50):
    mu=model.mu.detach();omega=model.log_omega.detach().exp();sigma=model.log_sigma.detach().exp();sy,sm=pool_support(st,sy,sm);n=len(events)
    eta=torch.zeros(n,2,dtype=mu.dtype,requires_grad=True);best_eta=eta.detach().clone();best=torch.full((n,),float('inf'),dtype=mu.dtype);fallback=torch.zeros(n,dtype=torch.bool)
    with torch.enable_grad():
        opt=torch.optim.Adam([eta],lr=.05)
        for step in range(steps+1 if sm.any() else 1):
            th=mu[None]+torch.nn.functional.pad(eta,(1,2));pred=floor_curve(th,events,st)
            objective=.5*torch.where(sm,((pred-sy)/sigma).square(),0.).sum(1)+.5*(eta/omega).square().sum(1)
            finite=torch.isfinite(objective);better=finite&(objective.detach()<best);best=torch.where(better,objective.detach(),best);best_eta=torch.where(better[:,None],eta.detach(),best_eta)
            if step==steps or not sm.any():break
            if not finite.all():fallback|=~finite;break
            opt.zero_grad();objective.sum().backward()
            if not torch.isfinite(eta.grad).all():fallback[:]=True;break
            torch.nn.utils.clip_grad_norm_([eta],20);opt.step()
            with torch.no_grad():eta.clamp_(-3,3)
    with torch.no_grad():
        th=mu[None]+torch.nn.functional.pad(best_eta,(1,2));y=floor_curve(th,events,qt)
    return y,dict(eta=best_eta,objective=best,fallback=fallback,parameters=th)
