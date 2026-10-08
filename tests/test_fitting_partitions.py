import numpy as np
import pandas as pd
import pytest
from mhap_pk.data.fitting_partitions import matched_partitions,validate_partition

def fixture():
 rows=[dict(source_id='s',participant_id=str(i),drug_id=d) for i in range(8) for d in ['a','b']]
 m=pd.DataFrame(rows);return m,(np.arange(12),np.arange(12,14),np.arange(14,16))

def test_matched_nested_and_linked():
 m,roles=fixture();previous=set()
 for cap in [2,4,6]:
  a=matched_partitions(m,roles,'a',cap,1,False);b=matched_partitions(m,roles,'a',cap,1,True)
  ta,va,ea=validate_partition(m,roles,a);tb,vb,eb=validate_partition(m,roles,b)
  assert set(ta)==set(tb[m.iloc[tb].drug_id.to_numpy()=='a'])
  assert previous<=set(ta);previous=set(ta)
  assert len(ta)==cap and np.array_equal(va,vb) and np.array_equal(ea,eb)

def test_reject_changed_evaluation_and_partial_linkage():
 m,roles=fixture();s=matched_partitions(m,roles,'a',2,1,True)
 s['train']=s['train'][:-1]
 with pytest.raises(ValueError,match='Partial'):validate_partition(m,roles,s)
 s=matched_partitions(m,roles,'a',2,1,True);s['test']=[15]
 with pytest.raises(ValueError,match='fixed'):validate_partition(m,roles,s)
