"""Experimental log-amount extended Kalman observer, not calibrated uncertainty."""
import torch
from ...state_ode import advance,transition
from ...relational import floor_curve,floor_concentration
from .joint_estimator import block_support

def covariance_filter(theta,events,st,sy,mask,qt,tau=.3,noise=.3,process=.01,mode='full',strength=1.):
    if mode not in ['full','central']:raise ValueError(mode)
    if tau<=0 or noise<=0 or process<0 or strength<0:raise ValueError('invalid covariance scales')
    st,sy,mask,_,first=block_support(st,sy,mask,events)
    cutoff=torch.where(mask,st,-torch.inf).max(1).values
    if not torch.isfinite(qt).all() or (qt<=cutoff[:,None]).any():raise ValueError('queries must follow support')
    # Positive-coordinate reference offset, not extra drug in the ODE.
    eps=theta[:,2].exp()[:,None]*1e-5
    origin=torch.minimum(torch.where(events[...,1]>0,events[...,0],torch.inf).min(1).values,first)
    state=advance(theta,torch.zeros_like(theta[:,:3]),origin,first[:,None],events,True)[:,0]
    active=theta.new_tensor([0.,1.,0.] if mode=='central' else [1.,1.,1.])
    covariance=torch.diag_embed(active[None].expand(len(theta),-1)*(tau**2)*strength)
    eye=torch.eye(3,dtype=theta.dtype,device=theta.device)[None]
    time=first;history=[];nll=[];clipped=[];gains=[];innov=[]
    for j in range(st.shape[1]):
        valid=mask[:,j];end=torch.where(valid,st[:,j],time);dt=(end-time).clamp_min(0)
        if j:
            before=advance(theta,state,time,end[:,None],events,False)[:,0]
            # Jacobian of log(A_next+eps) wrt log(A+eps), including known doses.
            F=transition(theta,dt)*(state+eps)[:,None,:]/(before+eps)[:,:,None]
            covariance=F@covariance@F.transpose(1,2)+torch.diag_embed(active[None]*((process**2)*dt*strength)[:,None])
            state=before
        if mode=='central':covariance=covariance*active[None,:,None]*active[None,None,:]
        prediction=floor_concentration(theta,state[:,None])[:,0]
        central=state[:,1];h=(central+eps[:,0])/central.clamp_min(eps[:,0]);h=torch.where(central>eps[:,0],h,torch.zeros_like(h))
        H=torch.stack([torch.zeros_like(h),h,torch.zeros_like(h)],-1)
        ph=(covariance@H[:,:,None]).squeeze(-1);S=(H*ph).sum(-1)+noise**2
        K=ph/S[:,None];residual=sy[:,j]-prediction;delta=K*residual[:,None]
        bounded=delta.clamp(-1.3862943611198906,1.3862943611198906)
        updated=((state+eps)*bounded.exp()-eps).clamp_min(0)
        M=eye-K[:,:,None]*H[:,None,:]
        post=M@covariance@M.transpose(1,2)+(noise**2)*K[:,:,None]*K[:,None,:]
        post=.5*(post+post.transpose(1,2))
        state=torch.where(valid[:,None],updated,state);covariance=torch.where(valid[:,None,None],post,covariance);time=end
        history.append(covariance);nll.append(torch.where(valid,.5*(S.log()+residual.square()/S),0.));clipped.append(valid&(delta.abs()>1.3862943611198906).any(1));gains.append(K);innov.append(residual)
    y=floor_concentration(theta,advance(theta,state,time,qt,events,False));base=floor_curve(theta,events,qt)
    y=torch.where(mask.any(1)[:,None],y,base)
    if strength==0:y=base
    return dict(mean=y,state=state,covariance=torch.stack(history,1),innovation_nll=torch.stack(nll,1),mask=mask,clipped=torch.stack(clipped,1),gains=torch.stack(gains,1),innovation=torch.stack(innov,1))
