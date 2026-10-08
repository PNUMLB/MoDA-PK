"""Cross-fitted scalar gates; exact prediction, cached training objectives.

The gate learns from out-of-fold future targets, never from outer test outcomes.
Output loss is analytic. Dynamics loss uses a uniform alpha grid for training;
deployment always propagates the exact ODE at the learned alpha.
"""
import numpy as np
import torch
from torch import nn
from .soft_hybrid import mix_dynamics
from .state_ode import advance
from .relational import floor_concentration

FEATURE_SETS = {'full':list(range(9)), 'no_distance':list(range(1,9)),
                'distance_only':[0], 'constant':[]}


def crossfit_splits(meta, fitting):
    """Two participant folds + three disjoint source-group folds, deterministic."""
    fitting=np.asarray(fitting,dtype=int)
    part={}
    for source in sorted(meta.iloc[fitting].source_id.unique()):
        people=sorted(meta.iloc[fitting][meta.iloc[fitting].source_id==source].participant_id.unique())
        part.update({p:i%2 for i,p in enumerate(people)})
    sources=sorted(meta.iloc[fitting].source_id.unique())
    source_group={s:i%3 for i,s in enumerate(sources)}
    episodes=[]
    for kind,n in [('known',2),('unseen',3)]:
        assignment=np.array([part[meta.iloc[i].participant_id] if kind=='known' else source_group[meta.iloc[i].source_id] for i in fitting])
        for group in range(n):
            train=fitting[assignment!=group];held=fitting[assignment==group]
            assert len(train) and len(held)
            assert set(meta.iloc[train].participant_id).isdisjoint(meta.iloc[held].participant_id)
            if kind=='unseen': assert set(meta.iloc[train].drug_id).isdisjoint(meta.iloc[held].drug_id)
            else: assert set(meta.iloc[held].drug_id)<=set(meta.iloc[train].drug_id)
            episodes.append(dict(name=f'{kind}_{group}',kind=kind,fitting_indices=train.tolist(),heldout_indices=held.tolist()))
    for kind in ['known','unseen']:
        assert sorted(i for e in episodes if e['kind']==kind for i in e['heldout_indices'])==sorted(fitting)
    return episodes


class ReliabilityGate(nn.Module):
    def __init__(self, mean, sd, feature_set='full'):
        super().__init__()
        self.register_buffer('feature_mean',mean.clone())
        self.register_buffer('feature_sd',sd.clamp_min(.1).clone())
        mask=torch.zeros_like(mean);mask[FEATURE_SETS[feature_set]]=1
        self.register_buffer('feature_mask',mask)
        self.linear=nn.Linear(9,1).to(dtype=mean.dtype,device=mean.device)
        nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)

    def forward(self, features):
        x=((features-self.feature_mean)/self.feature_sd).clamp(-8,8)*self.feature_mask
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def predict_cached(cache,alpha,fusion):
    """Exact query prediction from support-conditioned, frozen expert states."""
    if fusion=='output':
        return (1-alpha[:,None])*cache['seq_prediction']+alpha[:,None]*cache['info_prediction']
    theta,state=mix_dynamics(cache['sequential_parameters'],cache['sequential_state'],
        cache['information_parameters'],cache['information_state'],alpha)
    amounts=advance(theta,state,cache['time'],cache['query_t'],cache['events'],cache['include_start'])
    return floor_concentration(theta,amounts)


def profile_loss(prediction,cache):
    return torch.where(cache['query_mask'],(prediction-cache['query_y']).square(),0.).sum(1)/cache['query_mask'].sum(1)


def loss_grid(cache,nodes=65):
    values=[]
    with torch.no_grad():
        for a in torch.linspace(0,1,nodes,dtype=cache['features'].dtype):
            pred=predict_cached(cache,a.expand(len(cache['features'])),'dynamics')
            values.append(profile_loss(pred,cache))
    result=torch.stack(values,1)
    if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite dynamics loss grid')
    return result


def interpolate_loss(grid,alpha):
    position=alpha*(grid.shape[1]-1)
    lower=position.floor().long().clamp(0,grid.shape[1]-2)
    frac=position-lower
    row=torch.arange(len(alpha),device=alpha.device)
    return (1-frac)*grid[row,lower]+frac*grid[row,lower+1]


def fit_gate(features, weights, objective, feature_set='full', regularization=.01, steps=400):
    """Fixed optimization budget; no test or in-sample expert targets allowed.

    Caller supplies domain-normalized, source-balanced OOF query weights.
    Mean/sd use the original normalized OOF profile weights (not test features).
    """
    center=(features*weights[:,None]).sum(0)/weights.sum()
    sd=(((features-center).square()*weights[:,None]).sum(0)/weights.sum()).sqrt()
    gate=ReliabilityGate(center,sd,feature_set)
    optimizer=torch.optim.Adam(gate.parameters(),lr=.03)
    history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        a=gate(features)
        predictive=objective(a)
        penalty=regularization*gate.linear.weight.square().sum()
        loss=predictive+penalty
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite gate objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        if step%50==0 or step==steps-1:
            history.append(dict(step=step+1,loss=float(loss.detach()),predictive=float(predictive.detach()),penalty=float(penalty.detach())))
    return gate.eval(),history
