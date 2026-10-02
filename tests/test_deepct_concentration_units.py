import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch import nn
from mhap_pk.literature_adapters import DeepCtAdapter
class Constant(nn.Module):
 def __init__(self,p):super().__init__();self.register_buffer('p',p)
 def forward(self,z):return self.p[None].expand(len(z),-1)
def test_amount_volume_and_dose_scaling_for_all_routes():
 torch.set_num_threads(1)
 p=torch.tensor([1.,1.,10.,.3,15.,.2,20.,1.,10.,.3,15.,.2,20.],dtype=torch.double)
 old=DeepCtAdapter().double();new=DeepCtAdapter(concentration_convention='amount_over_volume').double()
 old.head=Constant(p);new.head=Constant(p);z=torch.zeros(1,512,dtype=torch.double);t=torch.tensor([[.5,1.,3.]],dtype=torch.double)
 for route,duration in [(0,0),(1,0),(2,2)]:
  e=torch.tensor([[[0.,100.,route,duration]]],dtype=torch.double)
  a=old.curve(z,e,t).exp();b=new.curve(z,e,t).exp()
  torch.testing.assert_close(a/b,torch.full_like(b,10. if route==0 else 1.))
  e2=e.clone();e2[:,:,1]*=2;torch.testing.assert_close(new.curve(z,e2,t).exp(),2*b)
 # Immediate IV bolus concentration is dose/Vc, independent of rates.
 e=torch.tensor([[[0.,100.,1.,0.]]],dtype=torch.double)
 torch.testing.assert_close(new.curve(z,e,torch.zeros(1,1,dtype=torch.double)).exp(),torch.tensor([[10.]],dtype=torch.double))

def test_shared_output_shift_does_not_make_unit_conventions_identical():
 torch.set_num_threads(1)
 p=torch.tensor([1.,1.,10.,.3,15.,.2,20.,1.,10.,.3,15.,.2,20.],dtype=torch.double)
 old=DeepCtAdapter().double();new=DeepCtAdapter(concentration_convention='amount_over_volume').double();old.head=Constant(p);new.head=Constant(p)
 z=torch.zeros(1,512,dtype=torch.double);e=torch.tensor([[[0.,100.,0.,0.]]],dtype=torch.double);st=torch.tensor([[.5,1.]],dtype=torch.double);qt=torch.tensor([[2.,3.]],dtype=torch.double);mask=torch.ones_like(st,dtype=torch.bool);sy=torch.zeros_like(st)
 difference=new(z,e,st,sy,mask,qt)['mean']-old(z,e,st,sy,mask,qt)['mean']
 # Two supplied observations partly compensate a constant route-scale change;
 # regularization leaves a known residual. Retrained results cannot be obtained
 # by merely dividing stored query predictions by volume.
 torch.testing.assert_close(difference,torch.full_like(difference,-.09/2.09*torch.log(torch.tensor(10.,dtype=torch.double))))
