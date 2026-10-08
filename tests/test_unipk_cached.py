import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pkforecast.literature_adapters import UniPKAdapter
from pkforecast.unipk_cached import CachedUniPKAdapter


class CachedUniPKTests(unittest.TestCase):
    def test_perturbed_nonlinear_dynamics_forecasts_and_gradients_match(self):
        torch.set_num_threads(1); torch.manual_seed(51)
        old, new = UniPKAdapter().double(), CachedUniPKAdapter().double()
        old.initialize(old.prior.base)
        with torch.no_grad():
            for p in old.parameters(): p.add_(torch.randn_like(p)*.005)
        new.load_state_dict(old.state_dict())
        z = torch.randn(3,512,dtype=torch.float64)
        ev = torch.tensor([[[0.,3.,0.,0.],[1.,2.,2.,1.]],
                           [[0.,4.,1.,0.],[2.,1.,1.,0.]],
                           [[0.,1.,2.,.3],[1.,2.,0.,0.]]],dtype=torch.float64)
        qt = torch.tensor([[.2,.7,1.5,4.],[.1,1.,2.5,5.],[.2,1.,2.,3.]],dtype=torch.float64)
        outputs=[]; grads=[]
        for model in [old,new]:
            output=model.curve(z,ev,qt)
            output.square().mean().backward()
            outputs.append(output.detach())
            grads.append({n:p.grad for n,p in model.named_parameters()})
        torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-7,atol=1e-7)
        for name in grads[0]:
            torch.testing.assert_close(grads[0][name],grads[1][name],rtol=2e-5,atol=1e-6)


if __name__ == '__main__': unittest.main()
