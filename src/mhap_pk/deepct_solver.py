"""DeepCt-inspired event solver with exact linear-compartment matrix exponential.
Official 13-parameter head order retained: PO(ka,CL,V1,Q2,V2,Q3,V3), IV(CL,V1,Q2,V2,Q3,V3).
This extends single-dose analytic curves to repeated and infusion events.
Uses physical amounts (umol), volumes (L), rates (L/hour), time (hour).
"""
import torch

def event_curves(params,events,times):
    if params.shape!=(13,) or not torch.isfinite(params).all() or (params<=0).any():
        raise ValueError('Expected 13 finite positive parameters')
    t=params.new_tensor([float(x) for x in times])
    if not torch.isfinite(t).all() or not len(t):raise ValueError('Invalid query times')
    result=torch.zeros_like(t)
    for e in events:
        if e.route=='oral':ka,cl,v1,q2,v2,q3,v3=params[:7]
        else:
            cl,v1,q2,v2,q3,v3=params[7:];ka=params.new_tensor(1.)
        zero=params.new_tensor(0.)
        k=torch.stack([torch.stack([-(cl+q2+q3)/v1,q2/v2,q3/v3,ka]),torch.stack([q2/v1,-q2/v2,zero,zero]),torch.stack([q3/v1,zero,-q3/v3,zero]),torch.stack([zero,zero,zero,-ka])])
        elapsed=(t-e.time).clamp_min(0)
        initial=params.new_zeros(4);initial[3 if e.route=='oral' else 0]=e.amount
        if e.route=='iv_infusion':
            forcing=params.new_zeros(4);forcing[0]=e.amount/e.duration
            augmented=torch.cat([torch.cat([k,forcing[:,None]],dim=1),params.new_zeros(1,5)],dim=0)
            s0=params.new_zeros(5);s0[-1]=1
            on=elapsed.clamp_max(e.duration)
            delivered=(torch.matrix_exp(augmented[None]*on[:,None,None])@s0)[:,:4]
            after=(elapsed-e.duration).clamp_min(0)
            states=(torch.matrix_exp(k[None]*after[:,None,None])@delivered[:,:,None]).squeeze(-1)
        else:states=torch.matrix_exp(k[None]*elapsed[:,None,None])@initial
        result=result+torch.where(t>=e.time,states[:,0]/v1,torch.zeros_like(t))
    return result
