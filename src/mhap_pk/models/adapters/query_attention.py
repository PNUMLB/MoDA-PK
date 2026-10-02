"""Query-time history readout with a bounded log-concentration residual.

This changes the observation/readout model, not compartment parameters. The
corrected output is not claimed to solve the original compartment equations.
"""
import math
import torch
from torch import nn
from .joint_estimator import JointPK
from ...relational import floor_curve

class QueryHistoryPK(nn.Module):
    def __init__(self, prior, kind='attention', bound=math.log(4)):
        super().__init__()
        if kind not in ['attention','mean']: raise ValueError(kind)
        self.backbone=JointPK(prior,mode='prior',full_sensitivity_features=True)
        self.kind,self.bound=kind,bound
        self.history=nn.Sequential(nn.Linear(47,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        # signed time, horizon, prior prediction, dose recency/amount/rate,
        # route indicators, and frozen molecular context.
        self.query=nn.Sequential(nn.Linear(41,32),nn.SiLU())
        if kind=='attention': self.cross=nn.MultiheadAttention(32,2,batch_first=True,dropout=0.)
        self.readout=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)

    def prepare(self,z,events,st,sy,sm,qt,ids=None):
        cutoff=torch.where(sm,st,-torch.inf).max(1).values
        if not torch.isfinite(qt).all() or (qt<=cutoff[:,None]).any():raise ValueError('Queries must follow support')
        p=self.backbone.prepare(z,events,st,sy,sm,ids)
        with torch.no_grad():
            base=floor_curve(p['mu'],events,qt)
            elapsed=qt[:,:,None]-events[:,None,:,0]
            past=(elapsed>=0)&(events[:,None,:,1]>0)
            infusion=events[:,None,:,2]==2;duration=events[:,None,:,3].clamp_min(1e-12)
            delivered=(events[:,None,:,1]*torch.where(infusion,(elapsed/duration).clamp(0,1),1.)*past).sum(-1)
            rate=(events[:,None,:,1]/duration*infusion*past*(elapsed<duration)).sum(-1)
            since=torch.where(past.any(-1),torch.where(past,elapsed,torch.inf).min(-1).values,0.)
            origin=torch.where(torch.isfinite(cutoff),cutoff,p['first'])
            routes=[(past&(events[:,None,:,2]==r)).any(-1) for r in range(3)]
            q=torch.stack([qt.sign()*qt.abs().log1p(),(qt-origin[:,None]).clamp_min(0).log1p(),base,since.log1p(),delivered.log1p(),rate.log1p(),*routes],-1)
            q=torch.cat([q,p['tokens'][:,:1,-32:].expand(-1,qt.shape[1],-1)],-1)
        return dict(tokens=p['tokens'],mask=p['sm'],query=q.detach(),base=base.detach(),support_times=p['st'])

    def forward(self,prepared):
        p=prepared;mask=p['mask'];h=self.history(p['tokens']);q=self.query(p['query'])
        if self.kind=='attention':
            safe=mask.clone();safe[~safe.any(1),0]=True
            context,weights=self.cross(q,h,h,key_padding_mask=~safe,need_weights=True,average_attn_weights=False)
            weights=weights*mask[:,None,None,:]
        else:
            weights=mask[:,None,None,:].to(h.dtype)/mask.sum(1).clamp_min(1)[:,None,None,None]
            weights=weights.expand(-1,1,q.shape[1],-1)
            context=(h*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
            context=context[:,None].expand(-1,q.shape[1],-1)
        delta=self.bound*torch.tanh(self.readout(torch.cat([q,context],-1)).squeeze(-1))
        delta=torch.where(mask.any(1)[:,None],delta,torch.zeros_like(delta))
        return dict(mean=p['base']+delta,correction=delta,attention=weights)
