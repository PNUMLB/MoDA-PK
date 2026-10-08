import torch
from mhap_pk.models.priors.hybrid_controls import build_control
from mhap_pk.models.priors.molecular_identity import build_joint_input

def test_full_factory_preserves_adopted_checkpoint_structure():
    torch.manual_seed(9);a=build_joint_input('molecular',3,torch.zeros(5,dtype=torch.double))
    torch.manual_seed(9);b=build_control('full',3,torch.zeros(5,dtype=torch.double))
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)

def test_branch_removals_ignore_only_registered_input():
    z=torch.randn(4,512,dtype=torch.double);i=torch.tensor([0,1,-1,2]);pop=torch.zeros(5,dtype=torch.double)
    a=build_control('without_molecule',3,pop)
    torch.testing.assert_close(a.prior(z,i)[1],a.prior(z+9,i)[1])
    b=build_control('without_identity',3,pop)
    torch.testing.assert_close(b.prior(z,i)[1],b.prior(z,torch.tensor([2,-1,0,1]))[1])
    assert a.prior.project[0].in_features==b.prior.project[0].in_features==1024

def test_update_switches_and_memoryless_dtype():
    pop=torch.zeros(5,dtype=torch.double)
    for variant,pa,st in [('without_updates',False,False),('parameter_only',True,False),('state_only',False,True)]:
        m=build_control(variant,3,pop);assert m.parameter_update==pa and m.state_update==st
    m=build_control('memoryless',3,pop);assert all(p.dtype==torch.double for p in m.parameters())
