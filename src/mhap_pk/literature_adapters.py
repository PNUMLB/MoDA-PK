"""Pinned official PK heads/dynamics with explicit forecasting adaptations.

Frozen UniMol inputs, event ledgers, consistent log floor, and support-only scale
MAP are shared adaptations. These are not unmodified end-to-end reproductions.
"""
from pathlib import Path
import importlib.util
import torch
from torch import nn
from torchdiffeq import odeint

VENDOR = Path(__file__).resolve().parents[2]/'third_party'


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Interface(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('base', torch.tensor([0., 0., 2.302585093, 0., 2.302585093]))

    def forward(self, z, ids=None):
        return self.base[None].expand(len(z), -1), z[:, :32], z.new_zeros((len(z), 5))


def inverse_softplus(value):
    return value+torch.log(-torch.expm1(-value))


class DeepCtAdapter(nn.Module):
    def __init__(self, id_count=0, concentration_convention='legacy'):
        super().__init__(); self.prior = Interface()
        if concentration_convention not in ('legacy', 'amount_over_volume'):
            raise ValueError('Unknown concentration convention')
        self.concentration_convention = concentration_convention
        native = module_from_file('pinned_deepct', VENDOR/'DeepCt/DeepCt.py')
        self.head = native.DeepCt(global_feats=512, n_tasks=13, predictor_hidden_feats=128, num_layers=2)

    def initialize(self, base):
        ka, cl, vc, q, vp = base.exp()
        parameters = torch.stack([ka, cl, vc, q, vp, q, vp, cl, vc, q, vp, q, vp])
        with torch.no_grad():
            self.head.predict[-1].weight.zero_()
            self.head.predict[-1].bias.copy_(inverse_softplus(parameters))

    def curve(self, z, events, times):
        p = self.head(z).clamp_min(1e-8)
        result = torch.zeros_like(times)
        for e in range(events.shape[1]):
            event = events[:, e]; oral = event[:, 2] == 0
            v = torch.where(oral[:, None], p[:, :7], torch.cat([p.new_ones((len(p), 1)), p[:, 7:]], -1))
            ka, cl, vc, q2, v2, q3, v3 = v.unbind(-1)
            zero = torch.zeros_like(ka)
            matrix = torch.stack([
                torch.stack([-(cl+q2+q3)/vc, q2/v2, q3/v3, ka], -1),
                torch.stack([q2/vc, -q2/v2, zero, zero], -1),
                torch.stack([q3/vc, zero, -q3/v3, zero], -1),
                torch.stack([zero, zero, zero, -ka], -1)], -2)
            dt = (times-event[:, 0, None]).clamp_min(0)
            initial = p.new_zeros((len(p), 4))
            initial[:, 0] = event[:, 1]*(~oral); initial[:, 3] = event[:, 1]*oral
            state = (torch.matrix_exp(matrix[:, None]*dt[..., None, None]) @ initial[:, None, :, None]).squeeze(-1)
            if (event[:, 2] == 2).any():
                force = p.new_zeros((len(p), 4)); force[:, 0] = event[:, 1]/event[:, 3].clamp_min(1e-12)
                augmented = torch.cat([torch.cat([matrix, force[..., None]], -1), p.new_zeros((len(p), 1, 5))], -2)
                on = torch.minimum(dt, event[:, 3, None]); after = (dt-event[:, 3, None]).clamp_min(0)
                delivered = torch.matrix_exp(augmented[:, None]*on[..., None, None])[..., :4, 4]
                infused = (torch.matrix_exp(matrix[:, None]*after[..., None, None]) @ delivered[..., None]).squeeze(-1)
                state = torch.where((event[:, 2] == 2)[:, None, None], infused, state)
            # Historical runs keep their convention. Corrected human-unit runs
            # interpret all compartments as amounts and report central amount/Vc.
            divisor = vc if self.concentration_convention == 'amount_over_volume' else torch.where(oral, torch.ones_like(vc), vc)
            concentration = state[..., 0]/divisor[:, None]
            result = result+torch.where(times >= event[:, 0, None], concentration, 0.)
        return result.clamp_min(1e-5).log()

    def forward(self, z, events, st, sy, sm, qt, ids=None, variance=False):
        merged = torch.cat([st, qt], 1)
        y = self.curve(z, events, merged)
        shift = torch.where(sm, sy-y[:, :st.shape[1]], 0.).sum(1)/(sm.sum(1)+.09)
        return dict(mean=y[:, st.shape[1]:]+shift[:, None], state={})


class UniPKAdapter(nn.Module):
    def __init__(self, id_count=0):
        super().__init__(); self.prior = Interface()
        native = module_from_file('pinned_unipk', VENDOR/'UniPKModel/models/pkmodel.py')
        self.native = native.UniPKModel(num_cmpts=3, route='p.o.', method='NeuralODE')

    def initialize(self, base):
        ka, cl, vc, q, vp = base.exp()
        with torch.no_grad():
            for head, values in [(self.native.volumeD.net, vc[None]),
                                 (self.native.cmptmodel.net, torch.stack([cl, q/vc, q/vc, q/vp, q/vp])),
                                 (self.native.cmptmodel.poka, ka[None])]:
                head[-2].weight.zero_(); head[-2].bias.copy_(inverse_softplus(values))

    def curve(self, z, events, times, rtol=1e-5, atol=1e-7):
        # Official dynamics; ledger extension supplies bolus/depot/infusion inputs.
        query = sorted(set(times.detach().cpu().flatten().tolist()))
        event_cpu = events.detach().cpu().numpy()
        start = min(query[0], float(event_cpu[..., 0][event_cpu[..., 1] > 0].min()))
        stop = query[-1]
        boundaries = {start, stop}
        # Stop a molecular group after its final requested time. Its state has
        # no influence on another profile; this avoids integrating short-lived
        # drugs out to an unrelated drug's 144-hour follow-up.
        _, groups = torch.unique(z.detach(), dim=0, return_inverse=True)
        group_end = times.max(1).values.clone()
        for group in groups.unique():
            selected = groups == group
            group_end[selected] = group_end[selected].max()
        boundaries.update(group_end.detach().cpu().tolist())
        for es in event_cpu:
            for t, amount, route, duration in es:
                if amount > 0:
                    boundaries.update(x for x in [float(t), float(t+duration)] if start <= x <= stop)
        boundaries = sorted(boundaries)
        volume = self.native.volumeD(z).clamp_min(1e-8)
        state = z.new_zeros((4, len(z))); saved = {}
        for k, left in enumerate(boundaries):
            jump = z.new_zeros((4, len(z)))
            for es in range(events.shape[1]):
                ev = events[:, es]
                dose = ev[:, 1]*(ev[:, 0] == left)
                jump[0] += dose*(ev[:, 2] == 1)/volume
                jump[-1] += dose*(ev[:, 2] == 0)/volume
            state = state+jump
            if left in query: saved[left] = state[0]
            if k == len(boundaries)-1: break
            right = boundaries[k+1]
            ev = events
            active = (ev[..., 2] == 2)&(ev[..., 0] <= left)&(ev[..., 0]+ev[..., 3] > left)
            forcing = z.new_zeros((4, len(z)))
            forcing[0] = (ev[..., 1]/ev[..., 3].clamp_min(1e-12)*active).sum(1)/volume
            needed = (group_end > left)[None]
            # Bind interval-specific values: adjoint is not used.
            def rhs(t, state):
                return (self.native.cmptmodel(t, state, z, volume)+forcing)*needed
            inside = [t for t in query if left < t < right]
            sol = odeint(rhs, state, z.new_tensor([left]+inside+[right]), rtol=rtol, atol=atol,
                options={'norm': lambda x: torch.linalg.vector_norm(x, dim=0).max()/2,
                         'max_num_steps': 10000})
            for j, t in enumerate(inside, 1): saved[t] = sol[j, 0]
            state = sol[-1]
        unique = torch.stack([saved[t] for t in query], -1)
        positions = torch.searchsorted(times.new_tensor(query), times)
        return unique.gather(1, positions).clamp_min(1e-5).log()

    def forward(self, z, events, st, sy, sm, qt, ids=None, variance=False):
        values = self.curve(z, events, torch.cat([st, qt], 1))
        past, prediction = values[:, :st.shape[1]], values[:, st.shape[1]:]
        shift = torch.where(sm, sy-past, 0.).sum(1)/(sm.sum(1)+.09)
        return dict(mean=prediction+shift[:, None], state={})
