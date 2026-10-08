import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torchdiffeq import odeint
from mhap_pk.unipk_cached import CachedUniPKAdapter
from mhap_pk.literature_adapters import UniPKAdapter

def test_cached_native_rhs_and_single_dose_units():
 torch.set_num_threads(1);torch.manual_seed(1)
 m=CachedUniPKAdapter().double();m.initialize(torch.tensor([0.,0.,2.3,0.,2.3],dtype=torch.double))
 ref=UniPKAdapter().double();ref.load_state_dict(m.state_dict())
 z=torch.randn(2,512,dtype=torch.double)*.1;t=torch.tensor([[0.,.2,1.,3.],[0.,.2,1.,3.]],dtype=torch.double)
 e=torch.tensor([[[0.,10.,1.,0.]],[[0.,10.,0.,0.]]],dtype=torch.double)
 torch.testing.assert_close(m.curve(z,e,t),ref.curve(z,e,t),rtol=1e-8,atol=1e-8)
 v=m.native.volumeD(z);s=torch.zeros(4,2,dtype=torch.double);s[0,0]=10/v[0];s[3,1]=10/v[1]
 native=odeint(lambda ti,y:m.native.cmptmodel(ti,y,z,v),s,t[0],rtol=1e-8,atol=1e-10)[:,0].T.clamp_min(1e-5).log()
 torch.testing.assert_close(m.curve(z,e,t,rtol=1e-8,atol=1e-10),native,rtol=1e-5,atol=1e-6)

def test_support_changes_only_explicit_scalar_correction():
 torch.set_num_threads(1);m=CachedUniPKAdapter().double();m.initialize(torch.tensor([0.,0.,2.3,0.,2.3],dtype=torch.double))
 z=torch.zeros(1,512,dtype=torch.double);e=torch.tensor([[[0.,10.,1.,0.]]],dtype=torch.double)
 st=torch.tensor([[.2,1.]],dtype=torch.double);qt=torch.tensor([[2.,3.]],dtype=torch.double);mask=torch.ones_like(st,dtype=torch.bool);sy=torch.zeros_like(st)
 a=m(z,e,st,sy,mask,qt)['mean'];b=m(z,e,st,sy+1,mask,qt)['mean']
 torch.testing.assert_close(b-a,torch.full_like(a,2/2.09))
