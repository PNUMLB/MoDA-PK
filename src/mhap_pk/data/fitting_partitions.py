"""Explicit reduced fitting partitions; the registered evaluation roles never change."""
import numpy as np
from mhap_pk.data.low_data import subset

def matched_partitions(meta, original, drug, cap, seed, pooled):
    train, valid, test = original
    kept, sampling = subset(meta, train, [drug], cap, seed)
    target = lambda ix: ix[meta.iloc[ix].drug_id.to_numpy() == drug]
    return dict(train=(kept if pooled else target(kept)).tolist(),
                validation=target(valid).tolist(), test=target(test).tolist(),
                target_drug=drug, fit_scope='donor_augmented' if pooled else 'target_only',
                sampling=sampling)

def validate_partition(meta, original, spec):
    roles = ('train', 'validation', 'test')
    result=[]
    for role, allowed in zip(roles,original):
        ix=np.asarray(spec[role],dtype=int)
        if not len(ix) or len(set(ix)) != len(ix) or not set(ix).issubset(set(allowed)):
            raise ValueError(f'Invalid {role} fitting indices')
        result.append(ix)
    drug=spec['target_drug']
    for ix, allowed in zip(result[1:],original[1:]):
        expected=allowed[meta.iloc[allowed].drug_id.to_numpy()==drug]
        if not np.array_equal(ix,expected):raise ValueError('Target validation/test must remain fixed')
    groups=[set(map(tuple,meta.iloc[ix][['source_id','participant_id']].to_numpy())) for ix in result]
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
        raise ValueError('Participant leakage')
    if spec['fit_scope']=='donor_augmented':
        selected=set(result[0])
        for _,g in meta.iloc[original[0]].groupby(['source_id','participant_id']):
            if len(set(g.index)&selected) not in (0,len(g)):
                raise ValueError('Partial linked participant in donor training')
    elif set(meta.iloc[result[0]].drug_id)!={drug}:
        raise ValueError('Target-only training contains another drug')
    return tuple(result)
