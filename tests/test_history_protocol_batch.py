import numpy as np
import torch
from mhap_pk.data.core import Batch
from mhap_pk.evaluation.history_protocols import common_origin
from mhap_pk.baselines.low_data import build

def batch():
 t=torch.tensor([[[0.,1.,1.,2.,3.]]*4],dtype=torch.double);y=torch.tensor([[[0.,.2,.3,.4,.5]]*4],dtype=torch.double);mask=torch.ones_like(t,dtype=torch.bool)
 return Batch(np.array([0]),torch.zeros(1,512,dtype=torch.double),torch.tensor([-1]),torch.tensor([[[0.,10.,1.,0.]]],dtype=torch.double),t,y,mask,torch.tensor([[4.,5.]],dtype=torch.double),torch.zeros(1,2,dtype=torch.double),torch.ones(1,2,dtype=torch.bool),torch.ones(1,2,dtype=torch.double)/2)

def test_common_origin_retains_replicates_and_identical_query():
 b=batch();c=common_origin(b)
 assert torch.equal(b.query_t,c.query_t) and torch.equal(b.query_y,c.query_y)
 assert [int(c.support_mask[0,i].sum()) for i in range(4)]==[0,1,3,5]
 assert torch.equal(c.support_t[0,2,c.support_mask[0,2]],torch.tensor([1.,1.,3.],dtype=torch.double))
 assert b.support_mask.all() # No mutation of first-k batch.

def test_hidden_support_labels_do_not_change_prediction():
 torch.manual_seed(7);c=common_origin(batch());m=build('molecular',2,torch.tensor([0.,0.,2.,0.,2.],dtype=torch.double)).eval();args=list(c.inputs(torch.tensor([2])))
 with torch.no_grad():a=m(*args)['mean'];args[3]=torch.where(args[4],args[3],torch.full_like(args[3],17));b=m(*args)['mean']
 torch.testing.assert_close(a,b,rtol=0,atol=0)

def test_unseen_offset_not_selected_from_any_training_drug():
 m=build('molecular_offset',2,torch.zeros(5,dtype=torch.double));z=torch.zeros(1,512,dtype=torch.double);ids=torch.tensor([-1])
 with torch.no_grad():
  a=m.prior(z,ids)[0];m.prior.offset.weight.fill_(10);b=m.prior(z,ids)[0]
 torch.testing.assert_close(a,b,rtol=0,atol=0)
