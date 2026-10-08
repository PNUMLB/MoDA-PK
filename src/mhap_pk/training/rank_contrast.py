"""Rank-N-Contrast-style auxiliary regression loss for PK query features.

Reference: Zha et al., NeurIPS 2023; official loss.py in
https://github.com/kaiwenzha/Rank-N-Contrast . This is a PK-task use of the
ranking objective, not a reproduction of their datasets/encoder/training.
"""
import torch
from torch import nn


def rank_contrast(features, labels, temperature=2.):
    if temperature <= 0 or len(features) < 2:
        raise ValueError('Need positive temperature and at least two samples')
    n=len(features);valid=~torch.eye(n,dtype=torch.bool,device=features.device)
    scores=-torch.cdist(features,features,p=2,compute_mode='donot_use_mm_for_euclid_dist')/temperature
    distances=(labels[:,None]-labels[None,:]).abs().detach()
    # i,j,k: candidates k at least as far in label space as positive j.
    eligible=(distances[:,None,:]>=distances[:,:,None]) & valid[:,None,:]
    denominator=torch.logsumexp(scores[:,None,:].expand(n,n,n).masked_fill(~eligible,-torch.inf),dim=-1)
    return (denominator-scores)[valid].mean()


class QueryProjection(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(77,32),nn.SiLU(),nn.Linear(32,16))

    def forward(self, model, batch, output, rows, cols):
        st=output['state'];q=batch.query_t[rows,cols]
        context=model.prior(batch.z,batch.ids)[1][rows]
        state=st['state'][rows].clamp_min(0).log1p()
        time=torch.stack([(q-st['time'][rows]).clamp_min(0).log1p(),q.sign()*q.abs().log1p()],-1)
        events=batch.events[rows];elapsed=q[:,None]-events[:,:,0];past=(elapsed>=0)&(events[:,:,1]>0);iv=events[:,:,2]==2
        delivered=events[:,:,1]*torch.where(iv,(elapsed/events[:,:,3].clamp_min(1e-12)).clamp(0,1),1.)*past
        rate=(events[:,:,1]/events[:,:,3].clamp_min(1e-12)*iv*past*(elapsed<events[:,:,3])).sum(1)
        since=torch.where(past.any(1),torch.where(past,elapsed,torch.inf).min(1).values,0.)
        dosing=torch.stack([delivered.sum(1).log1p(),rate.log1p(),since.log1p()],-1)
        feature=torch.cat([context,st['hidden'][rows],st['parameters'][rows],state,time,dosing],-1)
        return self.network(feature)


def sampled_query_indices(batch,generator,maximum=32):
    n=min(maximum,len(batch.indices))
    rows=torch.multinomial(batch.weights.sum(1),n,replacement=False,generator=generator)
    cols=torch.tensor([int(torch.where(batch.query_mask[r])[0][torch.randint(int(batch.query_mask[r].sum()),(),generator=generator)]) for r in rows])
    return rows,cols
