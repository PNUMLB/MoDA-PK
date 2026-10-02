"""Outcome-blind participant-linked Known-drug resampling; legacy splits untouched."""
import hashlib,json
import numpy as np

ROLES=('train','validation','test')
def partition_check(meta,spec,require_all=True):
    roles=[np.array(spec[r],dtype=int) for r in ROLES]
    for ix in roles:
        if not len(ix) or len(set(ix))!=len(ix) or min(ix)<0 or max(ix)>=len(meta):raise ValueError('Invalid partition indices')
    if set(roles[0])&set(roles[1]) or set(roles[0])&set(roles[2]) or set(roles[1])&set(roles[2]):raise ValueError('Profile overlap')
    if require_all and set(np.concatenate(roles))!=set(range(len(meta))):raise ValueError('Missing profiles')
    groups=[set(map(tuple,meta.iloc[ix][['source_id','participant_id']].astype(str).to_numpy())) for ix in roles]
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:raise ValueError('Linked participant overlap')
    for ix in roles:
        if set(meta.iloc[ix].drug_id)!=set(meta.drug_id):raise ValueError('Known-drug role lacks a drug')
    return tuple(roles)

def make_partitions(meta,split_seed=20261002):
    # Drug-connected components within each source keep crossover/coadministration groups intact.
    strata=[]
    for source,frame in meta.groupby('source_id',sort=True):
        groups=[(str(pid),list(g.index),set(g.drug_id)) for pid,g in frame.groupby('participant_id',sort=True)]
        drugs=sorted(frame.drug_id.unique());parent={d:d for d in drugs}
        def root(d):
            while parent[d]!=d:d=parent[d]
            return d
        for _,_,ds in groups:
            first=sorted(ds)[0]
            for d in ds:parent[root(d)]=root(first)
        bins={}
        for pid,ix,ds in groups:bins.setdefault(root(sorted(ds)[0]),[]).append((pid,ix))
        for component,items in sorted(bins.items()):
            if len(items)<5:raise ValueError('Fewer than five linked participants in a stratum; register separate protocol')
            key=json.dumps([source,component]);seed=int(hashlib.sha256(f'{split_seed}:{key}'.encode()).hexdigest()[:16],16)
            rng=np.random.default_rng(seed);items=[items[j] for j in rng.permutation(len(items))];strata.append((key,items))
    result={f'cv5_f{i:02d}':{r:[] for r in ROLES} for i in range(5)}
    result.update({x:{r:[] for r in ROLES} for x in ['ratio_712','ratio_811']})
    for key,items in strata:
        n=len(items);nv=max(1,int(np.floor(.1*n+.5)));nt20=max(1,int(np.floor(.2*n+.5)));nt10=max(1,int(np.floor(.1*n+.5)))
        # Identical validation groups; 10%-test nested in 20%-test. Extra test groups become training in 811.
        val=set(range(nt20,nt20+nv))
        for name,nt in [('ratio_712',nt20),('ratio_811',nt10)]:
            for j,(_,ix) in enumerate(items):result[name]['test' if j<nt else 'validation' if j in val else 'train']+=ix
        for f in range(5):
            test=set(j for j in range(n) if j%5==f);remaining=[j for j in range(n) if j not in test]
            rng=np.random.default_rng(int(hashlib.sha256(f'{split_seed}:{key}:inner:{f}'.encode()).hexdigest()[:16],16));vv=set(rng.permutation(remaining)[:nv].tolist())
            for j,(_,ix) in enumerate(items):result[f'cv5_f{f:02d}']['test' if j in test else 'validation' if j in vv else 'train']+=ix
    for spec in result.values():
        for r in ROLES:spec[r]=sorted(spec[r])
        partition_check(meta,spec)
    test=[x for name,spec in result.items() if name.startswith('cv5') for x in spec['test']]
    if sorted(test)!=list(range(len(meta))):raise ValueError('Each profile must be evaluated exactly once per CV repeat')
    assert set(result['ratio_811']['test']).issubset(result['ratio_712']['test'])
    assert result['ratio_811']['validation']==result['ratio_712']['validation']
    return result
