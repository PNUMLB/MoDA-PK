"""No future labels and no splitting of same-time observation replicates."""
from types import SimpleNamespace
import torch
import pytest
from mhap_pk.evaluation.history_protocols import initial_observation_inputs

def batch():
 return SimpleNamespace(indices=[0],z=torch.zeros(1,2),events=torch.zeros(1,1,4),ids=torch.zeros(1,dtype=torch.long),support_t=torch.tensor([[[0.,0.,0.,0.,0.]]*3+[[1.,1.,3.,5.,9.]]]),support_y=torch.tensor([[[0.,0.,0.,0.,0.]]*3+[[2.,4.,6.,8.,999.]]]),support_mask=torch.tensor([[[False]*5]*3+[[True]*5]]),query_t=torch.tensor([[10.,12.]]),query_y=torch.tensor([[12345.,67890.]]),query_mask=torch.tensor([[True,True]]))

def test_three_time_blocks_keep_replicates_and_remove_fourth_label():
 b=batch();args=initial_observation_inputs(b,3)
 assert args[2].tolist()==[[1.,1.,3.,5.,0.]]
 assert args[3].tolist()==[[2.,4.,6.,8.,0.]]
 assert args[4].sum()==4
 assert torch.equal(args[5],b.query_t)
 b.query_y.fill_(-9999);b.support_y[0,3,4]=-99999
 args2=initial_observation_inputs(b,3)
 assert all(torch.equal(a,c) for a,c in zip(args,args2))

@pytest.mark.parametrize('k',[0,1,2,3,4])
def test_count_means_distinct_times(k):
 b=batch();a=initial_observation_inputs(b,k)
 assert len(torch.unique(a[2][a[4]]))==k
 assert not a[3][~a[4]].any()


def test_query_at_t4_rejected():
 b=batch();b.query_t[0,0]=9.
 with pytest.raises(ValueError):initial_observation_inputs(b,3)
