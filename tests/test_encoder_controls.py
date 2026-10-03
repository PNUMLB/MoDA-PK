import torch
from mhap_pk.models.adapters.encoder_controls import EncoderControlPK

def test_masked_history_and_reset_and_prior():
 torch.set_num_threads(1)
 z=torch.zeros(2,512,dtype=torch.double);events=torch.tensor([[[0,1,1,0]],[[0,1,1,0]]],dtype=torch.double);st=torch.tensor([[.2,.5],[.2,.5]],dtype=torch.double);sy=torch.zeros_like(st);mask=torch.tensor([[True,True],[False,False]]);qt=torch.ones(2,2,dtype=torch.double)*2
 for kind in ['lstm','last_mlp','deep_sets','attention']:
  torch.manual_seed(1);m=EncoderControlPK(kind).double().eval();a=m(z,events,st,sy,mask,qt)['mean'];b=m(z,events,st,sy+100*(~mask),mask,qt)['mean'];torch.testing.assert_close(a,b);c=m(z,events,st,sy,mask,qt)['mean'];torch.testing.assert_close(a,c);assert torch.isfinite(a).all();a.sum().backward();assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
