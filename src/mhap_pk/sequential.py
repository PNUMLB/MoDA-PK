"""Molecular PK prior + ODE propagation + recurrent observation assimilation.

State updates change an estimate, not physical drug administration. After the
last support block, parameters freeze and only planned dosing drives the ODE.
Same-time replicates are pooled once (mean log observation, count as a feature).
The first implementation predicts means; it makes no uncertainty claim.
"""
import math
import torch
from torch import nn
from .state_ode import advance, concentration
from .batched import ForecastModel


class MolecularPrior(nn.Module):
    def __init__(self, id_count=0, hidden=32, chemical_bound=3.):
        super().__init__()
        self.chemical_bound = chemical_bound
        self.base = nn.Parameter(torch.tensor([0., 0., math.log(10), 0., math.log(10)]))
        self.project = nn.Sequential(nn.Linear(512, hidden), nn.SiLU())
        self.residual = nn.Linear(hidden, 5)
        self.ids = nn.Embedding(id_count, 512) if id_count else None
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)

    def forward(self, z, drug_indices=None):
        if self.ids is not None:
            known = drug_indices >= 0
            z = torch.where(known[:, None], self.ids(drug_indices.clamp_min(0)), self.ids.weight.mean(0)[None])
        context = self.project(z)
        deviation = self.chemical_bound*torch.tanh(self.residual(context)/self.chemical_bound)
        return self.base[None]+deviation, context, deviation


class SequentialPK(nn.Module):
    def __init__(self, id_count=0, hidden=32, state_update=True, parameter_update=True,
                 memory=True, fixed_observer=False, prior_only=False, allow_initial_state=True):
        super().__init__()
        self.prior = MolecularPrior(id_count=id_count, hidden=hidden)
        self.state_update, self.parameter_update = state_update, parameter_update
        self.memory, self.fixed_observer, self.prior_only = memory, fixed_observer, prior_only
        self.allow_initial_state = allow_initial_state
        # y, prediction, innovation, interval, elapsed-dose, delivered-dose,
        # active infusion rate, replicate count, and three route flags.
        if not (fixed_observer or prior_only):
            self.gru = nn.GRUCell(11+hidden, hidden)
            self.parameter_head = nn.Linear(hidden, 2)
            self.gain_head = nn.Linear(hidden, 1)
            nn.init.zeros_(self.parameter_head.weight); nn.init.zeros_(self.parameter_head.bias)
            nn.init.zeros_(self.gain_head.weight); nn.init.zeros_(self.gain_head.bias)
        self.hidden, self.personal_bound = hidden, 1.

    def condition(self, z, events, support_t, support_y, mask, drug_indices=None):
        if z.ndim != 2 or z.shape[1] != 512 or not torch.isfinite(z).all():
            raise ValueError('Expected finite [batch,512] molecular vectors')
        if support_t.ndim != 2 or support_t.shape != support_y.shape or mask.shape != support_t.shape:
            raise ValueError('Aligned batched support times, labels and mask required')
        if not torch.isfinite(support_t[mask]).all() or not torch.isfinite(support_y[mask]).all():
            raise ValueError('Support must be finite')
        if (events[..., 1] < 0).any() or not (events[..., 1] > 0).any(1).all():
            raise ValueError('Each profile requires valid dose events')
        active_event = events[..., 1] > 0
        legal_route = (events[..., 2] == 0) | (events[..., 2] == 1) | (events[..., 2] == 2)
        valid_duration = torch.where(events[..., 2] == 2, events[..., 3] > 0, events[..., 3] == 0)
        if not torch.isfinite(events).all() or not (legal_route & valid_duration)[active_event].all():
            raise ValueError('Invalid route, dose time, or infusion duration')
        # Stable sorting groups identical timestamps before recurrent processing.
        order = torch.argsort(torch.where(mask, support_t, torch.inf), dim=1, stable=True)
        times, labels, valid = [torch.gather(v, 1, order) for v in [support_t, support_y, mask]]
        mu, context, chemical = self.prior(z, drug_indices)
        params = mu
        b = len(z)
        first_event = torch.where(events[..., 1] > 0, events[..., 0], torch.inf).min(1).values
        first_observation = torch.where(valid, times, torch.inf).min(1).values if times.shape[1] else first_event
        current_time = torch.minimum(first_event, first_observation)
        state = z.new_zeros((b, 3)); hidden = z.new_zeros((b, self.hidden))
        first_interval = torch.ones(b, dtype=torch.bool, device=z.device)
        offsets = z.new_zeros((b, 2)); gain = z.new_zeros(b)
        parameter_penalty = z.new_zeros(b); count = z.new_zeros(b)
        trace = []
        for i in range(0 if self.prior_only else times.shape[1]):
            active = valid[:, i] & ((i == 0) | (times[:, i] != times[:, max(i-1, 0)]))
            if self.prior_only:
                active = active & False
            same_time = valid & (times == times[:, i, None])
            n = same_time.sum(1).clamp_min(1)
            y = (torch.where(same_time, labels, 0.).sum(1)/n)
            end = torch.where(active, times[:, i], current_time)
            predicted_state = advance(params, state, current_time, end[:, None], events, first_interval)[:, 0]
            pred = concentration(params, predicted_state)
            dt = (end-current_time).clamp_min(0)
            elapsed = end[:, None]-events[..., 0]
            past = (elapsed >= 0) & (events[..., 1] > 0)
            infusion = events[..., 2] == 2
            delivered = events[..., 1]*torch.where(infusion, (elapsed/events[..., 3].clamp_min(1e-12)).clamp(0, 1), 1.)*past
            rate = (events[..., 1]/events[..., 3].clamp_min(1e-12)*infusion*past*(elapsed < events[..., 3])).sum(1)
            since_last = torch.where(past.any(1), torch.where(past, elapsed, torch.inf).min(1).values, 0.)
            routes = [(past & (events[..., 2] == r)).any(1) for r in range(3)]
            innovation = y-pred
            features = torch.stack([y, pred, innovation, dt.log1p(), since_last.log1p(),
                                    delivered.sum(1).log1p(), rate.log1p(), n.log1p(), *routes], -1)
            if self.fixed_observer:
                proposed_h = hidden
                new_offset = torch.zeros_like(offsets)
                new_gain = torch.full_like(gain, .5)
            else:
                memory = hidden if self.memory else torch.zeros_like(hidden)
                proposed_h = self.gru(torch.cat([features, context], -1), memory)
                new_offset = self.personal_bound*torch.tanh(self.parameter_head(proposed_h))
                new_gain = torch.sigmoid(self.gain_head(proposed_h)).squeeze(-1)
            if not self.parameter_update:
                new_offset = new_offset*0
            change = torch.stack([torch.zeros_like(new_offset[:, 0]), new_offset[:, 0], new_offset[:, 1],
                                  torch.zeros_like(new_offset[:, 0]), torch.zeros_like(new_offset[:, 0])], -1)
            new_params = mu+change
            use_gain = new_gain if self.state_update else new_gain*0
            if not self.allow_initial_state:
                use_gain = torch.where(end == first_event, 0., use_gain)
            # Assimilate central concentration using a convex gain. Use the new
            # Vc to convert observation to amount, preserving nonnegative states.
            observed_c = (y.exp()-1e-5).clamp_min(0)
            observed_amount = observed_c*new_params[:, 2].exp()
            central = (1-use_gain)*predicted_state[:, 1]+use_gain*observed_amount
            proposed_state = torch.stack([predicted_state[:, 0], central, predicted_state[:, 2]], -1)
            state = torch.where(active[:, None], proposed_state, state)
            params = torch.where(active[:, None], new_params, params)
            hidden = torch.where(active[:, None], proposed_h, hidden)
            offsets = torch.where(active[:, None], new_offset, offsets)
            gain = torch.where(active, use_gain, gain)
            parameter_penalty = parameter_penalty+active*new_offset.square().mean(1)
            count = count+active
            current_time = end
            first_interval = first_interval & ~active
            trace.append(dict(time=end, active=active, log_prediction=pred, state=state, parameters=params, gain=use_gain))
        return dict(state=state, time=current_time, parameters=params, prior_parameters=mu, hidden=hidden,
                    include_start=first_interval, personal_offset=offsets, last_gain=gain,
                    regularization=parameter_penalty/count.clamp_min(1), chemical_offset=chemical, trace=trace)

    def forward(self, z, events, support_t, support_y, support_mask, query_t, drug_indices=None,
                variance=False):
        if query_t.ndim != 2 or query_t.shape[0] != len(z) or query_t.shape[1] == 0:
            raise ValueError('Expected nonempty batched query times')
        cutoff = torch.where(support_mask, support_t, -torch.inf).max(1).values if support_t.shape[1] else query_t.new_full((len(z),), -torch.inf)
        if not torch.isfinite(query_t).all() or (query_t <= cutoff[:, None]).any():
            raise ValueError('Queries must be finite and strictly later than all support')
        conditioned = self.condition(z, events, support_t, support_y, support_mask, drug_indices)
        states = advance(conditioned['parameters'], conditioned['state'], conditioned['time'], query_t,
                         events, conditioned['include_start'])
        prediction = concentration(conditioned['parameters'], states)
        return dict(mean=prediction, state=conditioned, regularization=conditioned['regularization'])


class MatchedInformationPK(ForecastModel):
    """The v4 local update using the identical new molecular prior, fixed SD=.6."""
    def __init__(self, id_count=0, hidden=32):
        nn.Module.__init__(self)
        self.prior = MolecularPrior(id_count=id_count, hidden=hidden)
        self.kind, self.fixed_std = 'v4', True
        self.register_buffer('noise', torch.tensor(.3, dtype=torch.float64))

    def distribution(self, z, drug_indices=None):
        mu, _, _ = self.prior(z, drug_indices)
        return mu, torch.full_like(mu, .6)
