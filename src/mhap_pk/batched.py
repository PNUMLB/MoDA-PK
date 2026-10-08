"""Vectorized v4 and matched ablations for complete cohort training.

The information solve/backtracking is identical to information.py. Jacobians
use exact forward-mode automatic differentiation through the analytic solver.
All labels in a call are support labels; future labels belong to the runner.
"""
import math
import torch
from torch import nn
from .kinetics import batch_curve, batch_jacobian


class ForecastModel(nn.Module):
    def __init__(self, kind='v4', hidden=32, fixed_std=False, id_count=0, noise=.3):
        super().__init__()
        self.kind, self.fixed_std = kind, fixed_std
        self.prior = nn.Sequential(nn.Linear(512, hidden), nn.SiLU(), nn.Linear(hidden, 10))
        self.register_buffer('noise', torch.tensor(noise, dtype=torch.float64))
        self.ids = nn.Embedding(id_count, 512) if id_count else None
        if kind == 'encoder':
            self.encoder = nn.Sequential(nn.Linear(11, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
            self.offset = nn.Sequential(nn.Linear(hidden + 5, hidden), nn.SiLU(), nn.Linear(hidden, 5))
            nn.init.zeros_(self.offset[-1].weight)
            nn.init.zeros_(self.offset[-1].bias)
        with torch.no_grad():
            self.prior[-1].weight.zero_()
            self.prior[-1].bias[:5].copy_(torch.tensor([0., 0., math.log(10), 0., math.log(10)]))
            self.prior[-1].bias[5:].fill_(math.log((.6 - .05) / (1.5 - .6)))

    def distribution(self, z, drug_indices=None):
        if self.ids is not None:
            known = drug_indices >= 0
            # An unseen identity has no trained embedding; use training-ID mean.
            z = torch.where(known[:, None], self.ids(drug_indices.clamp_min(0)), self.ids.weight.mean(0)[None])
        raw = self.prior(z)
        mu = raw[:, :5]
        std = torch.full_like(mu, .6) if self.fixed_std else .05 + 1.45 * torch.sigmoid(raw[:, 5:])
        return mu, std

    def condition(self, z, events, support_t, support_y, support_mask, drug_indices=None, mode=None):
        mu, std = self.distribution(z, drug_indices)
        b, p = mu.shape
        eye = torch.eye(p, dtype=mu.dtype, device=mu.device).expand(b, p, p)
        covariance = torch.diag_embed(std.square())
        if mode == 'prior' or self.kind == 'prior':
            return dict(mean=mu, covariance=covariance, prior_mean=mu, prior_std=std,
                        alpha=mu.new_zeros(b), information=eye * 0)
        if self.kind == 'encoder':
            base = batch_curve(mu, events, support_t)
            dt = support_t[:, :, None] - events[:, None, :, 0]
            past = (dt >= 0) & (events[:, None, :, 1] > 0)
            infusion = events[:, None, :, 2] == 2
            duration = events[:, None, :, 3].clamp_min(1e-12)
            amounts = events[:, None, :, 1] * torch.where(infusion, (dt / duration).clamp(0, 1), 1.)
            amount = (amounts * past).sum(-1)
            rate = (events[:, None, :, 1] / duration * infusion * past * (dt < duration)).sum(-1)
            last = torch.where(past, events[:, None, :, 0], -torch.inf).max(-1).values
            elapsed = torch.where(past.any(-1), support_t - last, 0.)
            count = past.sum(-1)
            routes = [(past & (events[:, None, :, 2] == i)).any(-1) for i in range(3)]
            features = torch.stack([support_t / 24, support_y, base, support_y - base,
                                    amount.log1p(), elapsed.log1p(), rate.log1p(), count.log1p(), *routes], -1)
            pooled = (self.encoder(features) * support_mask[..., None]).sum(1) / support_mask.sum(1).clamp_min(1)[:, None]
            delta = self.offset(torch.cat([pooled, mu], -1)).tanh() * support_mask.any(1)[:, None]
            # Matched log-parameter variant of v3; the archived v3 used softplus.
            return dict(mean=mu + delta, covariance=covariance, prior_mean=mu, prior_std=std,
                        alpha=mu.new_ones(b), information=eye * 0)
        base = batch_curve(mu, events, support_t)
        j = batch_jacobian(mu, events, support_t)
        a = j * std[:, None, :] / self.noise * support_mask[..., None]
        residual = (support_y - base) / self.noise * support_mask
        information = a.transpose(1, 2) @ a
        chol = torch.linalg.cholesky(eye + information)
        u = torch.cholesky_solve((a.transpose(1, 2) @ residual[..., None]), chol).squeeze(-1)
        white_cov = torch.cholesky_inverse(chol)
        covariance = std[:, :, None] * white_cov * std[:, None, :]
        shift = std * u
        with torch.no_grad():
            objectives = []
            sizes = [1., .5, .25, .125, 0.]
            for size in sizes:
                trial = mu + size * shift
                pred = batch_curve(trial, events, support_t)
                value = .5 * (((pred - support_y) / self.noise).square() * support_mask).sum(1)
                value = value + .5 * (size * u).square().sum(1)
                objectives.append(torch.where(torch.isfinite(value), value, torch.inf))
            chosen = torch.stack(objectives, -1).argmin(-1)
            alpha = mu.new_tensor(sizes)[chosen] * support_mask.any(1)
        return dict(mean=mu + alpha[:, None] * shift, covariance=covariance,
                    prior_mean=mu, prior_std=std, alpha=alpha, information=information)

    def forward(self, z, events, support_t, support_y, support_mask, query_t,
                drug_indices=None, mode=None, variance=True):
        state = self.condition(z, events, support_t, support_y, support_mask, drug_indices, mode)
        prediction = batch_curve(state['mean'], events, query_t)
        out = dict(mean=prediction, state=state)
        if variance:
            j = batch_jacobian(state['mean'], events, query_t)
            latent = (j @ state['covariance'] * j).sum(-1).clamp_min(0)
            out['variance'] = latent + self.noise.square()
        return out


def map_condition(model, z, events, support_t, support_y, support_mask, drug_indices=None, steps=50):
    """Inference-only MAP with exactly the same frozen prior/support objective.

    Fifty Adam steps in whitened coordinates, LR .05. Per-profile best objective
    includes the prior and local solution, so MAP never returns a worse objective.
    No query labels or validation residuals enter this fit.
    """
    with torch.no_grad():
        local = model.condition(z, events, support_t, support_y, support_mask, drug_indices)
        mu, std = local['prior_mean'].detach(), local['prior_std'].detach()
        start = (local['mean'].detach() - mu) / std
    def objective(u):
        prediction = batch_curve(mu + std * u, events, support_t)
        return .5 * (((prediction - support_y) / model.noise).square() * support_mask).sum(1) + .5 * u.square().sum(1)
    with torch.enable_grad():
        u = start.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([u], lr=.05)
        best_u, best = start.clone(), objective(start).detach()
        for _ in range(steps):
            optimizer.zero_grad()
            loss = objective(u)
            if not torch.isfinite(loss).all():
                raise FloatingPointError('Nonfinite MAP objective')
            loss.sum().backward()
            optimizer.step()
            with torch.no_grad():
                values = objective(u)
                better = torch.isfinite(values) & (values < best)
                best_u = torch.where(better[:, None], u, best_u)
                best = torch.minimum(values, best)
    result = {**local, 'mean': mu + std * best_u.detach(), 'map_objective': best,
              'map_steps': steps}
    return result
