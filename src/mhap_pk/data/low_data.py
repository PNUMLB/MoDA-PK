"""Nested target-only participant subsampling with connected coadministration blocks."""
import hashlib
import numpy as np

def drug_blocks(meta,train):
    d=meta.iloc[train];parent={x:x for x in sorted(d.drug_id.unique())}
    def root(x):
        while parent[x]!=x:x=parent[x]
        return x
    for _,g in d.groupby(['source_id','participant_id']):
        drugs=sorted(g.drug_id.unique())
        for x in drugs[1:]:parent[root(x)]=root(drugs[0])
    result={}
    for x in parent:result.setdefault(root(x),[]).append(x)
    return sorted([sorted(v) for v in result.values()])

def subset(meta,train,drugs,cap,seed):
    d=meta.iloc[train];target=d[d.drug_id.isin(drugs)]
    people=sorted(set(zip(target.source_id,target.participant_id)))
    salt=int(hashlib.sha256('|'.join(sorted(drugs)).encode()).hexdigest()[:8],16)
    order=np.random.default_rng(seed+salt).permutation(len(people));chosen=set(people[i] for i in order[:cap])
    target_people=set(people)
    keep=[int(i) for i in train if (meta.iloc[i].source_id,meta.iloc[i].participant_id) not in target_people or (meta.iloc[i].source_id,meta.iloc[i].participant_id) in chosen]
    # Every recorded profile of a selected/omitted person moves as a unit.
    selected=set(keep)
    for _,g in d.groupby(['source_id','participant_id']):assert len(set(g.index)&selected) in [0,len(g)]
    return np.array(keep),dict(target_drugs=drugs,participant_cap=cap,available_participants=len(people),chosen_participants=[list(x) for x in sorted(chosen)])
