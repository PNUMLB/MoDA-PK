import copy
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pkforecast.relational import build_relations, relation_loss, floor_curve
from pkforecast.relational_model import RelationalPK
from pkforecast.literature_adapters import DeepCtAdapter, UniPKAdapter
from pkforecast.deepct_solver import event_curves
from pkforecast.events import DoseEvent


class RelationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(11)
        self.profiles = []
        for d in range(4):
            self.profiles.append(dict(drug_id=str(d), participant_id=str(d),
                events=np.array([[0., 2., 0., 0.]]), query=pd.DataFrame(dict(
                    time=[1., 2., 4., 8.], target_log_dv=np.array([1., .5, 0., -.5])+d,
                    observation_id=[f'{d}_{t}' for t in range(4)]))))
        self.meta = pd.DataFrame(dict(drug_id=[str(d) for d in range(4)]))
        self.z = torch.randn(4, 512, dtype=torch.float64)
        self.ids = torch.arange(4)

    def build(self, profiles):
        return build_relations(profiles, self.meta, np.arange(3), self.z, self.ids)

    def test_heldout_labels_and_dose_cannot_change_fitting_relations(self):
        before = self.build(self.profiles)
        poisoned = copy.deepcopy(self.profiles)
        poisoned[3]['query'].target_log_dv = 1e9
        poisoned[3]['events'][:] = 1e9
        after = self.build(poisoned)
        for key in ['pk_distance', 'chemical_distance', 'pk_mask']:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertEqual(before['drugs'], ['0', '1', '2'])
        self.assertFalse(any(x.startswith('3_') for x in before['contributing_query_ids']))

    def test_dose_normalization_retains_pk_scale_and_same_route_requirement(self):
        before = self.build(self.profiles)
        self.assertAlmostEqual(float(before['pk_distance'][0, 1]), 1.)
        changed = copy.deepcopy(self.profiles)
        changed[1]['events'][0, 1] *= 10
        changed[1]['query'].target_log_dv += np.log(10)
        torch.testing.assert_close(before['pk_distance'], self.build(changed)['pk_distance'])
        changed[1]['events'][0, 2] = 1
        self.assertFalse(bool(self.build(changed)['pk_mask'][0, 1]))

    def test_no_time_overlap_does_not_become_negative(self):
        changed = copy.deepcopy(self.profiles)
        changed[1]['query'].time += 100
        self.assertFalse(bool(self.build(changed)['pk_mask'][0, 1]))

    def test_masked_relations_have_finite_gradients_and_ignore_unconnected_drug(self):
        z = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        mask = ~torch.eye(4, dtype=torch.bool); mask[3] = False; mask[:, 3] = False
        distance = torch.cdist(torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64))
        loss = relation_loss(z, distance, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad[:3].abs().sum()), 0)
        self.assertEqual(float(z.grad[3].abs().sum()), 0)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(8)
        self.model = RelationalPK().double()
        self.z = torch.randn(2, 512, dtype=torch.float64)
        self.ev = torch.tensor([[[0., 2., 0., 0.], [1., 1., 2., 1.]]]*2, dtype=torch.float64)
        self.st = torch.tensor([[.2, .5], [.2, .5]], dtype=torch.float64)
        self.sy = torch.tensor([[-1., -.8], [-.5, -.1]], dtype=torch.float64)
        self.sm = torch.ones_like(self.st, dtype=torch.bool)
        self.qt = torch.tensor([[2., 4.], [2., 4.]], dtype=torch.float64)

    def test_empty_history_is_consistent_floor_ode(self):
        out = self.model(self.z, self.ev, self.st, self.sy, ~self.sm, self.qt)
        mu = self.model.prior(self.z)[0]
        torch.testing.assert_close(out['mean'], floor_curve(mu, self.ev, self.qt))

    def test_future_queries_doses_and_masked_observations_cannot_change_past(self):
        out = self.model(self.z, self.ev, self.st, self.sy, self.sm, self.qt)
        q = torch.tensor([[4., 2., 3.], [4., 2., 3.]], dtype=torch.float64)
        ev = torch.cat([self.ev, torch.tensor([[[10., 4., 1., 0.]]]*2, dtype=torch.float64)], 1)
        st = torch.cat([self.st, torch.full((2, 1), 8., dtype=torch.float64)], 1)
        sy = torch.cat([self.sy, torch.full((2, 1), 1e6, dtype=torch.float64)], 1)
        sm = torch.cat([self.sm, torch.zeros((2, 1), dtype=torch.bool)], 1)
        other = self.model(self.z, ev, st, sy, sm, q)
        torch.testing.assert_close(out['mean'], other['mean'][:, [1, 0]], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(out['state']['state'], other['state']['state'], rtol=0, atol=0)

    def test_assimilation_link_and_gradients(self):
        out = self.model(self.z, self.ev, self.st, self.sy, self.sm, self.qt)
        (out['mean'].square().mean()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        first = out['state']['trace'][0]
        p = first['parameters']
        predicted = floor_curve(out['state']['prior_parameters'], self.ev, self.st[:, :1]).exp()[:, 0]
        expected = .5*predicted*p[:, 2].exp()+.5*self.sy[:, 0].exp()*p[:, 2].exp()
        torch.testing.assert_close(first['state'][:, 1], expected)

    def test_official_adapters_gradient_and_solver_agreement(self):
        for cls in [DeepCtAdapter, UniPKAdapter]:
            model = cls().double()
            model.initialize(model.prior.base)
            out = model(self.z, self.ev, self.st, self.sy, self.sm, self.qt)
            self.assertTrue(torch.isfinite(out['mean']).all())
            out['mean'].square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            if cls is UniPKAdapter:
                with torch.no_grad():
                    base = model.curve(self.z, self.ev, self.qt)
                    tight = model.curve(self.z, self.ev, self.qt, rtol=1e-8, atol=1e-10)
                torch.testing.assert_close(base, tight, rtol=0, atol=2e-4)


if __name__ == '__main__': unittest.main()
