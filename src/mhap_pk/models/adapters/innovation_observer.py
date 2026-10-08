"""Bounded log-innovation central-state observer; parameters fixed during replay."""
import math
import torch
from torch import nn
from .joint_estimator import JointPK
from ...state_ode import advance
from ...relational import floor_curve,floor_concentration

class InnovationObserver(nn.Module):
    def __init__(self,prior,learned=False,gain=.1,cl_update=False,memory="current"):
        super().__init__()
        if not 0<=gain<=1:raise ValueError('gain')
        self.backbone=JointPK(prior,mode='cl' if cl_update else 'prior',steps=10,full_sensitivity_features=True)
        if memory not in ['current','summary','gru','residual_mlp','residual_attention']:raise ValueError('memory')
        if learned and not 0<gain<1:raise ValueError('learned initial gain must be interior')
        self.memory=memory
        self.learned,self.gain,self.cl_update=learned,gain,cl_update
        if learned:
            if memory=='residual_mlp':
                self.controller=nn.Sequential(nn.Linear(44,16),nn.SiLU(),nn.Linear(16,1))
            elif memory=='residual_attention':
                self.token_project=nn.Sequential(nn.Linear(10,16),nn.SiLU())
                self.attention=nn.MultiheadAttention(16,2,batch_first=True,dropout=0.)
                self.controller=nn.Sequential(nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
            elif memory=='gru':
                self.recurrent=nn.GRUCell(8,16)
                self.controller=nn.Sequential(nn.Linear(16,1))
            else:
                self.controller=nn.Sequential(nn.Linear(15 if memory=='summary' else 8,16),nn.SiLU(),nn.Linear(16,1))
            nn.init.zeros_(self.controller[-1].weight)
            nn.init.constant_(self.controller[-1].bias,math.log(gain/(1-gain)))

    def prepare(self,z,events,st,sy,sm,ids=None):
        with torch.no_grad():
            p=self.backbone.prepare(z,events,st,sy,sm,ids)
            theta=self.backbone.infer(p)['parameters'] if self.cl_update else p['mu']
        return dict(theta=theta.detach(),events=events,st=p['st'],sy=p['sy'],mask=p['sm'])

    def forward(self,p,qt):
        theta=p['theta'];mask=p['mask'];events=p['events'];st=p['st'];cut=torch.where(mask,st,-torch.inf).max(1).values
        if not torch.isfinite(qt).all() or (qt<=cut[:,None]).any():raise ValueError('future queries required')
        firstevent=torch.where(events[...,1]>0,events[...,0],torch.inf).min(1).values
        start=torch.minimum(firstevent,st[:,0]);state=torch.zeros_like(theta[:,:3]);include=torch.ones_like(start,dtype=torch.bool);gains=[];residuals=[];corrections=[]
        hidden=theta.new_zeros((len(theta),16));previous=torch.zeros_like(start)
        count=torch.zeros_like(start);total=torch.zeros_like(start);total_abs=torch.zeros_like(start)
        feature_history=[];time_history=[];valid_history=[]
        if self.memory in ['residual_mlp','residual_attention'] and (mask.sum(1)>4).any():
            raise ValueError('Residual token experiment supports at most 4 time blocks')
        for j in range(st.shape[1]):
            valid=mask[:,j];end=torch.where(valid,st[:,j],start)
            before=advance(theta,state,start,end[:,None],events,include)[:,0]
            prediction=floor_concentration(theta,before[:,None])[:,0];residual=p['sy'][:,j]-prediction
            elapsed=end[:,None]-events[...,0];past=(elapsed>=0)&(events[...,1]>0)
            since=torch.where(past.any(1),torch.where(past,elapsed,torch.inf).min(1).values,0.)
            eps=theta[:,2].exp()*1e-5;sensitivity=before[:,1]/(before[:,1]+eps)
            rate=(theta[:,1]-theta[:,2]).exp()
            features=torch.stack([residual.clamp(-8,8),residual.abs().clamp_max(8),(end-start).clamp_min(0).log1p(),since.log1p(),sensitivity,(rate*(end-start).clamp_min(0)).log1p(),(past&(events[...,2]==0)).any(1),(past&(events[...,2]==2)).any(1)],-1)
            feature_history.append(features);time_history.append(end);valid_history.append(valid)
            if self.learned:
                if self.memory in ['residual_mlp','residual_attention']:
                    past_features=torch.stack(feature_history[-4:],1)
                    past_times=torch.stack(time_history[-4:],1)
                    past_mask=torch.stack(valid_history[-4:],1)
                    tokens=torch.cat([past_features,(past_times.sign()*past_times.abs().log1p())[:,:,None],
                                      (end[:,None]-past_times).clamp_min(0).log1p()[:,:,None]],-1)
                    tokens=torch.where(past_mask[:,:,None],tokens,torch.zeros_like(tokens))
                    if self.memory=='residual_mlp':
                        tokens=torch.cat([tokens,past_mask[:,:,None].to(tokens.dtype)],-1)
                        tokens=torch.nn.functional.pad(tokens,(0,0,0,4-tokens.shape[1]))
                        controller_input=tokens.flatten(1)
                    else:
                        embedded=self.token_project(tokens)
                        safe=past_mask.clone();safe[~safe.any(1),0]=True
                        query=embedded[:,-1:]
                        attended,_=self.attention(query,embedded,embedded,key_padding_mask=~safe,need_weights=False)
                        controller_input=torch.cat([query[:,0],attended[:,0]],-1)
                elif self.memory=='summary':
                    has_previous=count>0
                    difference=residual-previous
                    extra=torch.stack([previous.clamp(-8,8),difference.clamp(-8,8),
                        (difference/(end-start).clamp_min(1e-3)).clamp(-8,8),
                        (total/count.clamp_min(1)).clamp(-8,8),
                        (total_abs/count.clamp_min(1)).clamp_max(8),count.log1p(),has_previous],-1)
                    extra=torch.where(has_previous[:,None],extra,torch.zeros_like(extra))
                    controller_input=torch.cat([features,extra],-1)
                elif self.memory=='gru':
                    hidden=torch.where(valid[:,None],self.recurrent(features,hidden),hidden)
                    controller_input=hidden
                else:controller_input=features
                gain=self.controller(controller_input).squeeze(-1).sigmoid()
            else:gain=torch.full_like(start,self.gain)
            previous=torch.where(valid,residual,previous)
            total=total+torch.where(valid,residual,0.)
            total_abs=total_abs+torch.where(valid,residual.abs(),0.)
            count=count+valid
            update=gain*residual.clamp(-math.log(4),math.log(4));central=((before[:,1]+eps)*update.exp()-eps).clamp_min(0)
            after=torch.stack([before[:,0],central,before[:,2]],-1)
            state=torch.where(valid[:,None],after,state);start=torch.where(valid,end,start);include=include&~valid
            gains.append(torch.where(valid,gain,0.));residuals.append(torch.where(valid,residual,0.));corrections.append(torch.where(valid,update,0.))
        states=advance(theta,state,start,qt,events,include);y=floor_concentration(theta,states);base=floor_curve(theta,events,qt)
        y=torch.where(mask.any(1)[:,None],y,base)
        if not self.learned and self.gain==0:y=base
        return dict(mean=y,gains=torch.stack(gains,1),innovations=torch.stack(residuals,1),updates=torch.stack(corrections,1),state=state,parameters=theta)
