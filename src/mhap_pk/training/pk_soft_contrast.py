"""Training-only supervised PK soft assignments; inspired by soft contrast,
not a reproduction of SoftCLT. Projection is discarded at inference.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class HistoryProjection(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Linear(32,16))
    def forward(self,x):return self.net(x)

def training_affinity(profiles, fitting, same_drug_only=False):
    n=len(fitting);distance=np.zeros((n,n));eligible=np.zeros((n,n),dtype=bool);curves=[];used=[]
    for idx in fitting:
        p=profiles[int(idx)];q=p['query'].groupby('time').target_log_dv.mean();e=p['events']
        if len(e)!=1 or len(q)<3:
            curves.append(None);continue
        times=q.index.to_numpy()-e[0,0];keep=times>0
        if keep.sum()<3:
            curves.append(None);continue
        curves.append((e[0,2],round(float(e[0,3]),8),times[keep],q.to_numpy()[keep]-np.log(e[0,1])))
        used.extend(p['query'].observation_id.astype(str).tolist())
    for i in range(n):
        if curves[i] is None:continue
        for j in range(i+1,n):
            if curves[j] is None or curves[i][:2]!=curves[j][:2]:continue
            pi,pj=profiles[int(fitting[i])],profiles[int(fitting[j])]
            if same_drug_only and pi['drug_id']!=pj['drug_id']:continue
            if (pi['source_id'],pi['participant_id'])==(pj['source_id'],pj['participant_id']):continue
            a,b=curves[i],curves[j];lo=max(a[2][0],b[2][0]);hi=min(a[2][-1],b[2][-1])
            if hi<2*lo:continue
            grid=np.geomspace(lo,hi,12);ya=np.interp(np.log(grid),np.log(a[2]),a[3]);yb=np.interp(np.log(grid),np.log(b[2]),b[3])
            value=.5*(ya.mean()-yb.mean())**2+.5*np.mean(((ya-ya.mean())-(yb-yb.mean()))**2)
            distance[i,j]=distance[j,i]=value;eligible[i,j]=eligible[j,i]=True
    scale=max(float(np.median(distance[eligible])),.01) if eligible.any() else 1.
    affinity=np.where(eligible,np.exp(-distance/scale),0.)
    audit=dict(fitting_indices=list(map(int,fitting)),contributing_query_ids=sorted(set(used)),eligible_ordered_pairs=int(eligible.sum()),distance_scale=scale,supervision='Training future query labels define fixed soft targets; never validation/test, not self-supervised.',distance='Half squared dose-normalized log level difference + half centered log shape MSE on 12 log-time points; same single route/duration; overlapping time range factor >=2.')
    return torch.tensor(affinity,dtype=torch.float64),audit

def soft_view_loss(left,right,groups,affinity,temperature=.2,soft_mass=.2):
    n=len(left)
    if right.shape!=left.shape or affinity.shape!=(n,n) or len(groups)!=n or temperature<=0 or not 0<=soft_mass<1:raise ValueError('Invalid soft contrast inputs')
    if n==0:return (left.sum()+right.sum())*0
    allowed=torch.tensor([[i==j or groups[i]!=groups[j] for j in range(n)] for i in range(n)],device=left.device)
    eye=torch.eye(n,dtype=left.dtype,device=left.device)
    similarities=affinity.detach().to(left).clamp_min(0)*allowed*(1-eye)
    logits=F.normalize(left,dim=-1)@F.normalize(right,dim=-1).T/temperature
    def directional(scores,sims,mask):
        mass=sims.sum(-1,keepdim=True);alpha=soft_mass*(mass>0).to(left.dtype)
        target=(1-alpha)*eye+alpha*sims/torch.where(mass>0,mass,torch.ones_like(mass))
        logp=scores.masked_fill(~mask,-torch.inf).log_softmax(-1)
        return -(target*torch.where(mask,logp,0.)).sum(-1).mean()
    return .5*(directional(logits,similarities,allowed)+directional(logits.T,similarities.T,allowed.T))
