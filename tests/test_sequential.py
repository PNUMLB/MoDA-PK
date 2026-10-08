"""Independent physical reference, event boundaries and causal filter contracts."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pkforecast.state_ode import transition, advance, concentration
from pkforecast.kinetics import batch_curve
from pkforecast.sequential import SequentialPK, MatchedInformationPK


def matrix_reference(x, t):
    ka, cl, vc, q, vp = x.exp().unbind()
    zero = x.new_zeros(())
    k = torch.stack([torch.stack([-ka, zero, zero]),
                     torch.stack([ka, -(cl+q)/vc, q/vp]),
                     torch.stack([zero, q/vc, -q/vp])])
    return torch.matrix_exp(k[None]*t[:, None, None])


class StateODETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(9)
        self.x = torch.tensor([[.1, -.5, 2., .3, 2.5]], dtype=torch.float64)

    def test_transition_and_derivatives_against_independent_matrix_exponential(self):
        for i in range(4):
            x = torch.randn(5, dtype=torch.float64, requires_grad=True)
            t = torch.tensor([0., 1e-4, .2, 2., 80.], dtype=x.dtype)
            ref = matrix_reference(x, t)
            pred = transition(x[None], t[None])[0]
            torch.testing.assert_close(pred, ref, rtol=1e-9, atol=1e-10)
            weights = torch.randn_like(ref)
            grad = torch.autograd.grad((pred*weights).sum(), x, retain_graph=True)[0]
            grad_ref = torch.autograd.grad((ref*weights).sum(), x)[0]
            torch.testing.assert_close(grad, grad_ref, rtol=1e-8, atol=1e-9)
            self.assertTrue((pred >= -1e-12).all())
            self.assertTrue((pred.sum(-2) <= 1+1e-10).all())

    def test_all_routes_match_original_concentration_solver(self):
        events = torch.tensor([[[0., 3., 0., 0.], [.5, 2., 1., 0.], [1., 5., 2., 2.], [5., 1., 0., 0.]]], dtype=self.x.dtype)
        times = torch.tensor([[0., .3, .5, 1., 2., 3., 4., 5., 12.]], dtype=self.x.dtype)
        states = advance(self.x, torch.zeros(1, 3, dtype=self.x.dtype), torch.zeros(1, dtype=self.x.dtype), times, events, True)
        torch.testing.assert_close(concentration(self.x, states), batch_curve(self.x, events, times), rtol=1e-9, atol=1e-9)

    def test_filter_boundary_does_not_redose_and_continues_infusion(self):
        events = torch.tensor([[[0., 3., 0., 0.], [1., 2., 1., 0.], [0., 5., 2., 3.]]], dtype=self.x.dtype)
        zero = torch.zeros(1, dtype=self.x.dtype); s = torch.zeros(1, 3, dtype=self.x.dtype)
        first = advance(self.x, s, zero, zero[:, None]+1, events, True)[:, 0]
        split = advance(self.x, first, zero+1, zero[:, None]+5, events)
        whole = advance(self.x, s, zero, zero[:, None]+5, events, True)
        torch.testing.assert_close(split, whole, rtol=1e-10, atol=1e-10)

    def test_coincident_absorption_eigenvalue_gradient(self):
        x = torch.zeros(5, dtype=torch.float64, requires_grad=True)
        with torch.no_grad(): x[0] = torch.tensor((3+5**.5)/2).log()
        t = torch.tensor([.1, 1., 20.], dtype=x.dtype)
        pred, ref = transition(x[None], t[None])[0], matrix_reference(x, t)
        torch.testing.assert_close(pred, ref, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(torch.autograd.grad(pred.sum(), x, retain_graph=True)[0],
                                   torch.autograd.grad(ref.sum(), x)[0], rtol=1e-8, atol=1e-9)


class SequentialContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(13)
        self.model = SequentialPK().double()
        torch.nn.init.normal_(self.model.parameter_head.weight, std=.05)
        self.z = torch.randn(1, 512, dtype=torch.float64)
        self.events = torch.tensor([[[0., 4., 0., 0.], [.5, 3., 2., 2.]]], dtype=self.z.dtype)
        self.t = torch.tensor([[0., .4, .9]], dtype=self.z.dtype)
        self.y = torch.tensor([[-1., -.5, -.8]], dtype=self.z.dtype)
        self.mask = torch.ones_like(self.t, dtype=torch.bool)
        self.q = torch.tensor([[3., 5.]], dtype=self.z.dtype)

    def predict(self, t=None, y=None, mask=None, q=None, events=None):
        return self.model(self.z, self.events if events is None else events, self.t if t is None else t,
                          self.y if y is None else y, self.mask if mask is None else mask, self.q if q is None else q)

    def test_zero_support_equals_exact_molecular_ode(self):
        empty = torch.empty((1, 0), dtype=self.z.dtype)
        out = self.predict(t=empty, y=empty, mask=empty.bool())
        mu, _, _ = self.model.prior(self.z)
        torch.testing.assert_close(out['mean'], batch_curve(mu, self.events, self.q), rtol=1e-10, atol=1e-10)

    def test_query_grid_cannot_change_conditioned_state_or_predictions(self):
        base = self.predict(); reordered = self.predict(q=torch.tensor([[5., 4., 3., 3.]], dtype=self.z.dtype))
        torch.testing.assert_close(base['mean'], reordered['mean'][:, [2, 0]], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(base['state']['state'], reordered['state']['state'], rtol=0, atol=0)
        with self.assertRaises(ValueError): self.predict(q=self.t)

    def test_future_observation_cannot_affect_earlier_state(self):
        prefix = self.predict(t=self.t[:, :2], y=self.y[:, :2], mask=self.mask[:, :2])
        whole = self.predict(y=self.y+torch.tensor([[0., 0., 5.]], dtype=self.z.dtype))
        torch.testing.assert_close(prefix['state']['state'], whole['state']['trace'][1]['state'], rtol=0, atol=0)

    def test_sorting_and_same_time_replicates_are_order_invariant(self):
        t = torch.tensor([[0., .4, .4, .9]], dtype=self.z.dtype)
        y = torch.tensor([[-1., -.3, -.7, -.8]], dtype=self.z.dtype)
        mask = torch.ones_like(t, dtype=torch.bool)
        a = self.predict(t=t, y=y, mask=mask)
        perm = [2, 3, 0, 1]
        b = self.predict(t=t[:, perm], y=y[:, perm], mask=mask[:, perm])
        torch.testing.assert_close(a['mean'], b['mean'], rtol=1e-12, atol=1e-12)
        self.assertEqual(sum(int(s['active'].sum()) for s in a['state']['trace']), 3)

    def test_future_dose_does_not_affect_past_filter_or_earlier_queries(self):
        extra = torch.tensor([[[10., 7., 1., 0.]]], dtype=self.z.dtype)
        first = self.predict(); other = self.predict(events=torch.cat([self.events, extra], 1))
        torch.testing.assert_close(first['mean'], other['mean'], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(first['state']['state'], other['state']['state'], rtol=0, atol=0)

    def test_positive_time_zero_can_update_state_and_only_cl_vc_change(self):
        out = self.predict(); state = out['state']
        self.assertGreater(float(state['trace'][0]['state'][0, 1].detach()), 0.)
        self.assertTrue((state['state'] >= 0).all())
        torch.testing.assert_close(state['parameters'][:, [0, 3, 4]], state['prior_parameters'][:, [0, 3, 4]], rtol=0, atol=0)
        self.assertTrue((state['personal_offset'].abs() <= 1).all())

    def test_gradient_reaches_molecular_branch_gru_and_both_heads(self):
        # Nonzero residual weights exercise molecular projection gradients.
        torch.nn.init.normal_(self.model.prior.residual.weight, std=.02)
        out = self.predict()
        (out['mean'].square().mean()+.001*out['regularization'].mean()).backward()
        for name in ['prior.project.0.weight', 'gru.weight_ih', 'parameter_head.weight', 'gain_head.weight']:
            gradient = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0., name)

    def test_information_control_has_identical_prior_initialization(self):
        torch.manual_seed(22); seq = SequentialPK().double()
        torch.manual_seed(22); control = MatchedInformationPK().double()
        for key, tensor in seq.prior.state_dict().items():
            torch.testing.assert_close(tensor, control.prior.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
