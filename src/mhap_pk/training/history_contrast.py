"""Same-cutoff two-view losses on observer memory; no query labels."""
import torch
import torch.nn.functional as F


def view_loss(left, right, groups, kind, temperature=.2):
    if left.shape != right.shape or left.ndim != 2 or len(groups) != len(left):
        raise ValueError('Aligned representations and participant groups required')
    if temperature <= 0 or kind not in ('consistency', 'contrastive'):
        raise ValueError('Invalid loss configuration')
    if len(left)==0:
        return (left.sum()+right.sum())*0
    a=F.normalize(left,dim=-1);b=F.normalize(right,dim=-1)
    if kind=='consistency':
        return (a-b).square().sum(-1).mean()
    # Cross-view positives on diagonal. Other profiles of the same participant
    # are neither negatives nor additional positives. No within-view negatives.
    allowed=torch.tensor([[i==j or groups[i]!=groups[j] for j in range(len(groups))]
                          for i in range(len(groups))],device=left.device)
    logits=(a@b.T)/temperature
    target=torch.arange(len(left),device=left.device)
    return .5*(F.cross_entropy(logits.masked_fill(~allowed,-torch.inf),target)
               +F.cross_entropy(logits.T.masked_fill(~allowed.T,-torch.inf),target))
