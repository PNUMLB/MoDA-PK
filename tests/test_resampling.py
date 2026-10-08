import pandas as pd
import pytest
from mhap_pk.data.resampling import make_partitions,partition_check

def fixture():
    rows=[]
    for i in range(6):
        for drug in ['a','b']:
            for occasion in range(2):rows.append(dict(source_id='s',participant_id=str(i),drug_id=drug,profile_id=f'{i}/{drug}/{occasion}'))
    for i in range(19):rows.append(dict(source_id='t',participant_id=str(i),drug_id='c',profile_id=str(i)))
    return pd.DataFrame(rows)
def test_linkage_and_oof():
    m=fixture();p=make_partitions(m);assert p==make_partitions(m)
    tests=[]
    for name,s in p.items():
        partition_check(m,s)
        if name.startswith('cv5'):tests+=s['test']
    assert sorted(tests)==list(range(len(m)))
    assert p['ratio_712']['validation']==p['ratio_811']['validation']
def test_detect_leakage():
    m=fixture();p=make_partitions(m)['ratio_712'];ix=next(i for i in p['test'] if m.iloc[i].source_id=='s');p['test'].remove(ix);p['train'].append(ix)
    with pytest.raises(ValueError):partition_check(m,p)
def test_small_stratum_explicit_failure():
    with pytest.raises(ValueError):make_partitions(fixture().iloc[:4])
