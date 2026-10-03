import pytest
import torch
from mhap_pk.models.priors.molecular_identity import MolecularIdentityPrior

def test_unseen_uses_only_training_id_mean_and_retains_molecular_input():
    torch.manual_seed(2)
    m=MolecularIdentityPrior(3).double()
    z=torch.randn(2,512,dtype=torch.double)
    out=m(z,torch.tensor([-1,-1]))
    expected=m.project(torch.cat((z,m.ids.weight.mean(0).expand(2,-1)),dim=-1))
    torch.testing.assert_close(out[1],expected)
    assert not torch.equal(out[1][0],out[1][1])

def test_both_branches_receive_gradients():
    torch.manual_seed(2)
    m=MolecularIdentityPrior(3).double()
    z=torch.randn(3,512,dtype=torch.double,requires_grad=True)
    m(z,torch.tensor([0,1,-1]))[1].square().sum().backward()
    assert z.grad.abs().sum()>0
    assert m.ids.weight.grad.abs().sum()>0
    assert m.project[0].weight.grad[:,:512].abs().sum()>0
    assert m.project[0].weight.grad[:,512:].abs().sum()>0

@pytest.mark.parametrize('ids',[None,torch.tensor([3]),torch.tensor([-2])])
def test_unregistered_identity_rejected(ids):
    with pytest.raises(ValueError):MolecularIdentityPrior(3).double()(torch.zeros(1,512,dtype=torch.double),ids)
