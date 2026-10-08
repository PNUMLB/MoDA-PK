"""Drug-separated, participant-balanced soft contrast; eligible drugs equal."""
import torch
import torch.nn.functional as F

def within_drug_loss(left,right,groups,drugs,affinity,temperature=.2,soft_mass=.2):
    n=len(left)
    if right.shape!=left.shape or affinity.shape!=(n,n) or len(groups)!=n or len(drugs)!=n or temperature<=0 or not 0<=soft_mass<1:raise ValueError('Invalid within-drug inputs')
    values=[];records=[]
    for drug in sorted(set(drugs)):
        idx=[i for i,d in enumerate(drugs) if d==drug];gs=[groups[i] for i in idx];people=list(dict.fromkeys(gs))
        if len(people)<2:
            records.append(dict(drug_id=drug,anchors=len(idx),participants=len(people),eligible=False,pairs=0,loss=0.));continue
        a=F.normalize(left[idx],dim=-1);b=F.normalize(right[idx],dim=-1);m=len(idx)
        allowed=torch.tensor([[i==j or gs[i]!=gs[j] for j in range(m)] for i in range(m)],device=left.device)
        eye=torch.eye(m,device=left.device,dtype=left.dtype)
        weights=left.new_tensor([1/(len(people)*gs.count(g)) for g in gs])
        sim=affinity[idx][:,idx].detach().to(left).clamp_min(0)*allowed*(1-eye)
        logits=a@b.T/temperature
        def direction(scores,s):
            mass=s.sum(1,keepdim=True);alpha=soft_mass*(mass>0).to(left.dtype)
            target=(1-alpha)*eye+alpha*s/torch.where(mass>0,mass,torch.ones_like(mass))
            lp=scores.masked_fill(~allowed,-torch.inf).log_softmax(-1)
            per=-(target*torch.where(allowed,lp,0.)).sum(-1)
            return (per*weights).sum()
        loss=.5*(direction(logits,sim)+direction(logits.T,sim.T));values.append(loss)
        records.append(dict(drug_id=drug,anchors=m,participants=len(people),eligible=True,pairs=int(allowed.sum())-m,loss=float(loss.detach())))
    return (torch.stack(values).mean() if values else (left.sum()+right.sum())*0),records
