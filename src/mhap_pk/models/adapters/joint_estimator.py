"""Differentiable penalized CL/Vc + first-support central-state inference.

A single parameter vector and a single initial-state correction define all
support and query predictions. The encoder controls two regularization weights,
not concentration. Frozen priors, first-support post-event convention, and
analytic dose-event dynamics are retained. No posterior calibration is claimed.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from ...state_ode import advance
from ...relational import floor_concentration, floor_curve


def _initial(theta, events, first, eta):
    start = torch.minimum(torch.where(events[..., 1] > 0, events[..., 0], torch.inf).min(1).values, first)
    a = advance(theta, torch.zeros_like(theta[:, :3]), start, first[:, None], events, True)[:, 0]
    offset = theta[:, 2].exp() * 1e-5
    central = ((a[:, 1] + offset) * eta.exp() - offset).clamp_min(0.)
    return torch.stack((a[:, 0], central, a[:, 2]), -1)


def trajectory(u, mu, events, first, times, active):
    """u=(delta_logCL,delta_logVc,log_offset_amount_ratio)."""
    u = u * active
    zeros = torch.zeros_like(u[:, 0])
    theta = mu + torch.stack((zeros, u[:, 0], u[:, 1], zeros, zeros), -1)
    a = _initial(theta, events, first, u[:, 2])
    # a includes all dose effects at first; continuation must not repeat bolus.
    states = advance(theta, a, first, times, events, False)
    return floor_concentration(theta, states), theta, a


def _single(u, mu, events, first, times, active):
    return trajectory(u[None], mu[None], events[None], first[None], times[None], active)[0][0]


sensitivity = torch.vmap(torch.func.jacfwd(_single, argnums=0), in_dims=(0, 0, 0, 0, 0, None))


def block_support(times, labels, mask, events):
    """One log-mean token per distinct time; count is a feature, not a weight."""
    rows = []
    for t, y, valid in zip(times, labels, mask):
        t, y = t[valid], y[valid]
        if not torch.isfinite(t).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite support')
        ts = torch.unique(t, sorted=True)
        rows.append((ts, torch.stack([y[t == a].mean() for a in ts]) if len(ts) else y,
                     torch.tensor([(t == a).sum() for a in ts], dtype=times.dtype, device=times.device)))
    width = max(1, max(len(r[0]) for r in rows))
    first_event = torch.where(events[..., 1] > 0, events[..., 0], torch.inf).min(1).values
    first = torch.stack([r[0][0] if len(r[0]) else first_event[i] for i, r in enumerate(rows)])
    st = first[:, None].expand(-1, width).clone(); sy = torch.zeros_like(st)
    counts = torch.zeros_like(st); sm = torch.zeros_like(st, dtype=torch.bool)
    for i, (t, y, n) in enumerate(rows):
        st[i, :len(t)], sy[i, :len(y)], counts[i, :len(n)] = t, y, n
        sm[i, :len(t)] = True
    return st, sy, sm, counts, first


class ObservationEncoder(nn.Module):
    def __init__(self, kind, input_size=47, hidden=32,
                 lambda_bounds=(.1, 10.), lambda_initial=(1., 1.)):
        super().__init__()
        if kind not in ['deepsets', 'gru', 'attention', 'cross_attention']:
            raise ValueError('Unknown observation encoder')
        if not 0 < lambda_bounds[0] < lambda_bounds[1]:
            raise ValueError('Invalid lambda bounds')
        if len(lambda_initial) != 2 or any(not lambda_bounds[0] <= v <= lambda_bounds[1] for v in lambda_initial):
            raise ValueError('Initial lambda outside bounds')
        self.kind = kind; self.hidden = hidden
        self.log_lambda_min = math.log(lambda_bounds[0])
        self.log_lambda_span = math.log(lambda_bounds[1]) - self.log_lambda_min
        self.project = nn.Sequential(nn.Linear(input_size, hidden), nn.SiLU())
        if kind == 'deepsets':
            self.transform = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        elif kind == 'gru':
            self.cell = nn.GRUCell(hidden, hidden)
        else:
            self.attn = nn.MultiheadAttention(hidden, 2, dropout=0., batch_first=True)
            self.norm1 = nn.LayerNorm(hidden); self.norm2 = nn.LayerNorm(hidden)
            self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))
        if kind == 'cross_attention':
            self.molecular_query = nn.Linear(32, hidden)
            self.cross = nn.MultiheadAttention(hidden, 2, dropout=0., batch_first=True)
            self.cross_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        fractions = torch.tensor([(math.log(v)-self.log_lambda_min)/self.log_lambda_span
                                  for v in lambda_initial]).clamp(1e-6, 1-1e-6)
        with torch.no_grad(): self.head.bias.copy_(torch.logit(fractions))

    def forward(self, tokens, mask):
        x = self.project(tokens)
        if self.kind == 'gru':
            h = x.new_zeros((len(x), self.hidden))
            for i in range(x.shape[1]):
                h = torch.where(mask[:, i, None], self.cell(x[:, i], h), h)
        else:
            if self.kind == 'deepsets':
                x = self.transform(x)
            else:
                safe = mask.clone(); safe[~safe.any(1), 0] = True
                y, _ = self.attn(x, x, x, key_padding_mask=~safe, need_weights=False)
                x = self.norm1(x + y); x = self.norm2(x + self.ff(x))
            if self.kind == 'cross_attention':
                # The last 32 features are the frozen prior's molecular context,
                # identical across tokens. Query contains no future observations.
                q = self.molecular_query(tokens[:, :1, -32:])
                pooled, _ = self.cross(q, x, x, key_padding_mask=~safe, need_weights=False)
                h = self.cross_norm(q + pooled)[:, 0]
                h = torch.where(mask.any(1)[:, None], h, torch.zeros_like(h))
            else:
                h = (x * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        # Bounded positive penalization; runtime bounds belong in the config.
        return (self.log_lambda_min + self.log_lambda_span * self.head(h).sigmoid()).exp()


class JointPK(nn.Module):
    def __init__(self, prior, encoder='fixed', steps=3, mode='joint', evidence=True,
                 fixed_strength=(1., 1.), noise=.3, parameter_scale=.6, state_scale=1.,
                 lambda_bounds=(.1, 10.), lambda_initial=(1., 1.), full_sensitivity_features=False):
        super().__init__()
        modes = {'prior': (0, 0, 0), 'cl': (1, 0, 0), 'parameter': (1, 1, 0),
                 'state': (0, 0, 1), 'joint': (1, 1, 1)}
        if mode not in modes or steps < 0:
            raise ValueError('Invalid inference configuration')
        if min(*fixed_strength, noise, parameter_scale, state_scale) <= 0:
            raise ValueError('Scales and regularization must be positive')
        self.prior = prior
        for p in self.prior.parameters(): p.requires_grad_(False)
        self.encoder_kind = encoder
        self.controller = None if encoder == 'fixed' else ObservationEncoder(
            encoder, lambda_bounds=lambda_bounds, lambda_initial=lambda_initial)
        self.steps, self.mode, self.evidence = steps, mode, evidence
        self.full_sensitivity_features = full_sensitivity_features
        self.register_buffer('active', torch.tensor(modes[mode], dtype=torch.float64))
        self.register_buffer('bounds', torch.tensor([2., 2., math.log(4.)], dtype=torch.float64))
        self.register_buffer('scales', torch.tensor([parameter_scale, parameter_scale, state_scale], dtype=torch.float64))
        self.register_buffer('fixed_strength', torch.tensor(fixed_strength, dtype=torch.float64))
        self.noise = noise

    def prepare(self, z, events, st, sy, sm, ids=None):
        if not torch.isfinite(events).all() or not (events[..., 1] > 0).any(1).all():
            raise ValueError('Finite positive dosing history required')
        if (events[..., 1] < 0).any(): raise ValueError('Negative dose')
        active = events[..., 1] > 0
        legal = ((events[..., 2] == 0) | (events[..., 2] == 1) | (events[..., 2] == 2))
        duration = torch.where(events[..., 2] == 2, events[..., 3] > 0, events[..., 3] == 0)
        if not (legal & duration)[active].all(): raise ValueError('Invalid dose route/duration')
        with torch.no_grad():
            mu, context, _ = self.prior(z, ids)
            st, sy, sm, count, first = block_support(st, sy, sm, events)
            u = mu.new_zeros((len(mu), 3))
            base = trajectory(u, mu, events, first, st, self.active)[0]
            feature_active = torch.ones_like(self.active) if self.full_sensitivity_features else self.active
            j = sensitivity(u, mu, events, first, st, feature_active)
            elapsed = st[:, :, None] - events[:, None, :, 0]
            past = (elapsed >= 0) & (events[:, None, :, 1] > 0)
            inf = events[:, None, :, 2] == 2
            duration = events[:, None, :, 3].clamp_min(1e-12)
            delivered = (events[:, None, :, 1] * torch.where(inf, (elapsed / duration).clamp(0, 1), 1.) * past).sum(-1)
            rate = (events[:, None, :, 1] / duration * inf * past * (elapsed < duration)).sum(-1)
            since = torch.where(past.any(-1), torch.where(past, elapsed, torch.inf).min(-1).values, 0.)
            dt = torch.cat([torch.zeros_like(st[:, :1]), (st[:, 1:] - st[:, :-1]).clamp_min(0.)], 1)
            route = [(past & (events[:, None, :, 2] == r)).any(-1) for r in range(3)]
            evidence = float(self.evidence)
            features = torch.stack([st.sign() * st.abs().log1p(), dt.log1p(), sy, base * evidence,
                                    (sy - base) * evidence, *(j * self.scales * evidence).unbind(-1),
                                    since.log1p(), delivered.log1p(), rate.log1p(), count.log1p(), *route], -1)
            tokens = torch.cat([features, context[:, None].expand(-1, st.shape[1], -1)], -1)
        if not torch.isfinite(tokens).all(): raise FloatingPointError('Nonfinite prepared features')
        return dict(mu=mu.detach(), events=events, st=st, sy=sy, sm=sm, first=first, tokens=tokens.detach())

    def infer(self, prepared, strengths=None):
        p = prepared; mu = p['mu']; mask = p['sm']; u = mu.new_zeros((len(mu), 3))
        lam = strengths if strengths is not None else (self.controller(p['tokens'], mask) if self.controller is not None else self.fixed_strength[None].expand(len(mu), -1))
        reg = torch.stack([lam[:, 0], lam[:, 0], lam[:, 1]], -1) / self.scales.square()
        def predict(v): return trajectory(v, mu, p['events'], p['first'], p['st'], self.active)[0]
        def objective(v):
            e = torch.where(mask, (predict(v) - p['sy']) / self.noise, 0.)
            return .5 * e.square().sum(1) + .5 * (reg * v.square()).sum(1)
        before = objective(u); history = [before]; fallback = torch.zeros(len(mu), dtype=torch.bool, device=mu.device)
        for _ in range(0 if self.mode == 'prior' else self.steps):
            pred = predict(u); j = sensitivity(u, mu, p['events'], p['first'], p['st'], self.active)
            good = torch.isfinite(pred).all(1) & torch.isfinite(j).all((1, 2))
            fallback = fallback | ~good
            j = torch.where(mask[..., None] & good[:, None, None], torch.nan_to_num(j), 0.) / self.noise
            residual = torch.where(mask & good[:, None], torch.nan_to_num(p['sy'] - pred), 0.) / self.noise
            h = j.transpose(1, 2) @ j + torch.diag_embed(reg + 1e-4)
            rhs = (j.transpose(1, 2) @ residual[..., None]).squeeze(-1) - reg * u
            step = torch.linalg.solve(h, rhs[..., None]).squeeze(-1)
            candidates = [torch.maximum(torch.minimum(u + a * step, self.bounds), -self.bounds) * self.active for a in [1., .5, .25, 0.]]
            costs = torch.stack([objective(v) for v in candidates], 1)
            finite = torch.isfinite(costs)
            choice = torch.where(finite, costs, torch.inf).detach().argmin(1)
            new = torch.stack(candidates, 1)[torch.arange(len(mu), device=mu.device), choice]
            good = good & torch.isfinite(new).all(1) & finite.any(1)
            fallback = fallback | ~good
            u = torch.where((good & mask.any(1))[:, None], new, torch.zeros_like(new))
            history.append(objective(u))
        # Any numerical failure in this episode conservatively reverts to prior.
        u = torch.where(fallback[:, None], torch.zeros_like(u), u)
        _, theta, state = trajectory(u, mu, p['events'], p['first'], p['st'], self.active)
        return dict(update=u, parameters=theta, first_state=state, first_time=p['first'],
                    strengths=lam, objective_before=before, objective_after=objective(u),
                    objective_history=torch.stack(history, 1), fallback=fallback)

    def forward(self, z, events, st, sy, sm, qt, ids=None, prepared=None):
        cutoff = torch.where(sm, st, -torch.inf).max(1).values if st.shape[1] else qt.new_full((len(qt),), -torch.inf)
        if not torch.isfinite(qt).all() or (qt <= cutoff[:, None]).any():
            raise ValueError('Queries must be finite and strictly after support')
        p = self.prepare(z, events, st, sy, sm, ids) if prepared is None else prepared
        state = self.infer(p)
        y = trajectory(state['update'], p['mu'], events, p['first'], qt, self.active)[0]
        # No-support episodes must use the original dosing initial condition,
        # including queries before first dose, not an artificial support origin.
        prior = floor_curve(p['mu'], events, qt)
        y = torch.where((p['sm'].any(1) & ~state['fallback'] & (self.mode != 'prior'))[:, None], y, prior)
        return dict(mean=y, state=state)
