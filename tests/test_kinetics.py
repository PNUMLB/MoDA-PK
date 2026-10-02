"""Independent matrix-exponential equivalence including first/second derivatives."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pkforecast.events import DoseEvent
from pkforecast.kinetics import curve_single, batch_jacobian, pack_events
from pkforecast.reference_solver import event_curves


class ExactSolutionTests(unittest.TestCase):
    def test_curves_and_derivatives_against_matrix_exponential(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        for route in ['oral', 'iv_bolus', 'iv_infusion']:
            events = [DoseEvent(0, 10, route, .5 if route == 'iv_infusion' else 0),
                      DoseEvent(2, 3, 'oral'), DoseEvent(10, 1, 'iv_bolus')]
            times = torch.tensor([-.1, 0, .01, .5, 1, 2, 2.1, 4, 12, 100.], dtype=torch.float64)
            for _ in range(3):
                x = torch.randn(5, dtype=torch.float64, requires_grad=True)
                ref = lambda v: torch.log(event_curves(v.exp(), events, times).clamp_min(0) + 1e-5)
                fun = lambda v: curve_single(v, pack_events(events), times)
                torch.testing.assert_close(fun(x), ref(x), atol=2e-9, rtol=2e-9)
                j = torch.autograd.functional.jacobian(fun, x, create_graph=True)
                jr = torch.autograd.functional.jacobian(ref, x, create_graph=True)
                torch.testing.assert_close(j, jr, atol=2e-7, rtol=2e-7)
                torch.testing.assert_close(torch.autograd.grad(j.square().sum(), x)[0],
                                           torch.autograd.grad(jr.square().sum(), x)[0], atol=2e-5, rtol=2e-5)

    def test_equal_absorption_and_disposition_rate(self):
        x = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float64).log()
        # alpha = (3+sqrt(5))/2; exercise the removable singularity exactly.
        x[0] = torch.log(torch.tensor((3 + 5 ** .5) / 2, dtype=x.dtype))
        events = [DoseEvent(0, 10, 'oral')]
        t = torch.tensor([0., .1, 1., 20.], dtype=x.dtype)
        out = curve_single(x, pack_events(events), t)
        ref = torch.log(event_curves(x.exp(), events, t).clamp_min(0) + 1e-5)
        torch.testing.assert_close(out, ref, atol=1e-9, rtol=1e-9)
        self.assertTrue(torch.isfinite(batch_jacobian(x[None], pack_events(events)[None], t[None])).all())


if __name__ == '__main__':
    unittest.main()
