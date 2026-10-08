from dataclasses import replace
import numpy as np,pandas as pd,torch
from mhap_pk.data.core import Batch
from mhap_pk.relational import floor_curve
from mhap_pk.training.synthetic_augmentation import generate

def test_synthetic_prefix_consistency_and_training_weights(tmp_path):
 torch.set_num_threads(1);pop=torch.tensor([0.,0.,2.,0.,2.5],dtype=torch.double);ev=torch.tensor([[[0.,10.,1.,0.]],[[0.,12.,1.,0.]]],dtype=torch.double);t=torch.tensor([[.2,.5,1.,2.]]*2,dtype=torch.double);qt=torch.tensor([[3.,4.]]*2,dtype=torch.double);mu=pop.repeat(2,1);hy=floor_curve(mu,ev,t);qy=floor_curve(mu,ev,qt);sm=torch.zeros(2,4,4,dtype=torch.bool)
 for i,k in enumerate([0,1,2,4]):sm[:,i,:k]=True
 b=Batch(np.array([0,1]),torch.zeros(2,512,dtype=torch.double),torch.zeros(2,dtype=torch.long),ev,t[:,None].repeat(1,4,1),hy[:,None].repeat(1,4,1),sm,qt,qy,torch.ones_like(qt,dtype=torch.bool),torch.ones_like(qt)/4)
 meta=pd.DataFrame({'drug_id':['a','a','held_out']});a=tmp_path/'a';a.mkdir();s=generate([],meta,np.array([0,1]),b,pop,1,a)
 torch.testing.assert_close(s.weights,b.weights);torch.testing.assert_close(s.query_t,b.query_t);torch.testing.assert_close(s.events,b.events)
 for ki in range(4):torch.testing.assert_close(s.support_y[:,ki],s.support_y[:,3])
 assert torch.isfinite(s.query_y).all();assert not torch.equal(s.query_y,b.query_y)

def test_centered_augmentation_removes_teacher_offset_not_perturbation(tmp_path):
 import json
 from mhap_pk.training.centered_augmentation import generate as centered
 torch.set_num_threads(1);mu=torch.tensor([[0.,0.,2.,0.,2.5]],dtype=torch.double);ev=torch.tensor([[[0.,10.,1.,0.]]],dtype=torch.double);t=torch.tensor([[.2,.5,1.,2.]],dtype=torch.double);qt=torch.tensor([[3.,4.]],dtype=torch.double)
 h0=floor_curve(mu,ev,t);q0=floor_curve(mu,ev,qt);sm=torch.zeros(1,4,4,dtype=torch.bool)
 for ki,k in enumerate([0,1,2,4]):sm[:,ki,:k]=True
 b=Batch(np.array([0]),torch.zeros(1,512,dtype=torch.double),torch.zeros(1,dtype=torch.long),ev,t[:,None].repeat(1,4,1),(h0+1.7)[:,None].repeat(1,4,1),sm,qt,q0+1.7,torch.ones_like(qt,dtype=torch.bool),torch.ones_like(qt)/2)
 source=tmp_path/'source';source.mkdir();out=tmp_path/'centered';out.mkdir();(source/'synthetic_generator.json').write_text(json.dumps({'fitting_indices':[0]}));torch.save(dict(teacher_parameters=mu,support=h0+.2,query=q0+.2),source/'synthetic_training_LOCAL_ONLY.pt')
 result=centered([],None,np.array([0]),b,mu[0],1,out,source)
 torch.testing.assert_close(result.query_y,b.query_y+.2)
 for ki in range(4):torch.testing.assert_close(result.support_y[:,ki],b.support_y[:,3]+.2)
 torch.testing.assert_close(result.weights,b.weights);torch.testing.assert_close(result.events,b.events)
 import pytest
 with pytest.raises(AssertionError):centered([],None,np.array([1]),b,mu[0],1,out,source)
