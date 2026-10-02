"""Support-only calibration contract; no query target access."""
import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).resolve().parents[1]/'scripts/run_scale_shape_diagnostic.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_scale_shrinkage_matches_closed_form():
 q=torch.tensor([[3.,4.]],dtype=torch.double);r=torch.tensor([[2.,2.]],dtype=torch.double)
 y,b,a=m.predict([(0.,q,r,torch.ones_like(r,dtype=torch.bool))],1.)
 assert torch.allclose(b,torch.tensor([4/3],dtype=torch.double))
 assert torch.allclose(y,q+4/3)
 assert a.item()==0

def test_no_history_keeps_prior():
 q=torch.tensor([[3.,4.]],dtype=torch.double);r=torch.zeros((1,2),dtype=torch.double);mask=torch.zeros_like(r,dtype=torch.bool)
 y,b,a=m.predict([(t,q,r,mask) for t in m.TIMES],.1,.3)
 assert torch.equal(y,q) and b.item()==0 and a.item()==0

def test_query_predictions_do_not_select_rate_or_scale():
 mask=torch.ones((1,2),dtype=torch.bool);c=[(a,torch.tensor([[1.,2.]],dtype=torch.double),torch.tensor([[a+1.,1.-a]],dtype=torch.double),mask) for a in m.TIMES]
 _,b,a=m.predict(c,.1,.3)
 _,b2,a2=m.predict([(t,q*1000,r,ms) for t,q,r,ms in c],.1,.3)
 assert torch.equal(b,b2) and torch.equal(a,a2)

def test_large_residual_scale_bound():
 q=torch.zeros((1,2),dtype=torch.double);r=q+100
 _,b,_=m.predict([(0.,q,r,torch.ones_like(r,dtype=torch.bool))],.1)
 assert b.item()==8
