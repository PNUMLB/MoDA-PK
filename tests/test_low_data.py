import numpy as np,pandas as pd,torch
from mhap_pk.data.low_data import drug_blocks,subset
from mhap_pk.models.priors.low_data import make_low_data_model

def test_linkage_nested_subsets_keep_donors_and_all_occasions():
 d=pd.DataFrame([dict(source_id='S',participant_id=str(i),drug_id=drug) for i in range(5) for drug in ['a','b']]+[dict(source_id='T',participant_id='donor',drug_id='c')])
 tr=np.arange(len(d));assert drug_blocks(d,tr)==[['a','b'],['c']]
 a,_=subset(d,tr,['a','b'],2,1);b,_=subset(d,tr,['a','b'],4,1)
 assert set(a)<set(b) and 10 in a and len(a)==5 and len(b)==9
 assert all(n in [0,2] for n in d.loc[a[a<10]].groupby('participant_id').size())

def test_rate_scale_has_same_initial_parameters_and_separates_volume_from_rates():
 t=torch.tensor([.2,1.1,2.4,.5,3.1],dtype=torch.double);z=torch.randn(3,512,dtype=torch.double);m=make_low_data_model('scale_rate',2,t);mu=m.prior(z)[0];torch.testing.assert_close(mu,t.expand(3,-1))
 with torch.no_grad():m.prior.scale.bias.fill_(.3)
 after=m.prior(z)[0];torch.testing.assert_close(after-mu,torch.tensor([0.,.3,.3,.3,.3],dtype=torch.double).expand_as(mu))

def test_unseen_drug_offset_exact_shared_fallback_and_identity_no_test_mapping():
 m=make_low_data_model('molecular_offset',2,[0.,1.,2.,1.,2.]);z=torch.randn(2,512,dtype=torch.double)
 with torch.no_grad():m.prior.offset.weight.fill_(.5)
 out=m.prior(z,torch.tensor([-1,0]))[0];base=m.prior.base[None]+m.prior.residual(m.prior.project(z));torch.testing.assert_close(out[0],base[0],atol=0,rtol=0);assert not torch.allclose(out[1],base[1])
