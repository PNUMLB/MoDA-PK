"""Information-conditioned molecular PK adaptation.

One local Gaussian update in log-parameter space, with optional support-only
backtracking of the mean. The covariance is a local linear approximation, not
an exact nonlinear posterior or a validated prediction interval.

The Gaussian update and sensitivity analysis are established tools. The research
candidate is their episodically trainable coupling to molecular priors for
cross-drug, sparse-history forecasting. No novelty or accuracy claim is encoded.
"""
import math
from pathlib import Path
import sys

import torch
from torch import nn

from .reference_solver import event_curves


def gaussian_information_update(mu, prior_std, support_mean, jacobian, observed, noise_std):
    """Exact conditioning of the *linearized* Gaussian observation model.

    The solve uses prior-whitened coordinates. Unobserved directions in these
    coordinates retain zero mean shift and unit variance. For a diagonal prior,
    a zero Jacobian column retains that parameter's prior mean and variance.
    """
    p = mu.numel()
    n = observed.numel()
    if mu.ndim != 1 or prior_std.shape != mu.shape or jacobian.shape != (n, p):
        raise ValueError('Invalid parameter/Jacobian shapes')
    if observed.ndim != 1 or support_mean.shape != observed.shape:
        raise ValueError('Support predictions and observations must align')
    sigma = torch.as_tensor(noise_std, dtype=mu.dtype, device=mu.device).expand_as(observed)
    if any(not torch.isfinite(x).all() for x in [mu, prior_std, support_mean, jacobian, observed, sigma]):
        raise ValueError('Nonfinite conditioning inputs')
    if (prior_std <= 0).any() or (sigma <= 0).any():
        raise ValueError('Standard deviations must be positive')
    eye = torch.eye(p, dtype=mu.dtype, device=mu.device)
    if n == 0:
        return dict(mean=mu, covariance=torch.diag(prior_std.square()),
                    whitened_shift=torch.zeros_like(mu), whitened_covariance=eye,
                    information=eye * 0, gains=torch.zeros_like(mu), effective_rank=0)
    a = jacobian * prior_std[None, :] / sigma[:, None]
    residual = (observed - support_mean) / sigma
    information = a.T @ a
    chol = torch.linalg.cholesky(eye + information)
    shift = torch.cholesky_solve((a.T @ residual)[:, None], chol).squeeze(1)
    whitened_covariance = torch.cholesky_inverse(chol)
    covariance = prior_std[:, None] * whitened_covariance * prior_std[None, :]
    # Eigenvectors never enter the differentiated solve; repeated eigenvalues
    # therefore do not cause SVD/eigendecomposition gradient singularities.
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvalsh(information).clamp_min(0).flip(0)
        gains = eigenvalues / (1 + eigenvalues)
        rank = int(torch.linalg.matrix_rank(a, atol=1e-9, rtol=1e-7))
    return dict(mean=mu + prior_std * shift, covariance=covariance,
                whitened_shift=shift, whitened_covariance=whitened_covariance,
                information=information, gains=gains, effective_rank=rank)


class InformationConditionedPK(nn.Module):
    def __init__(self, embedding_dim=512, hidden=32, prior_std_min=.05,
                 prior_std_max=1.5, noise_std=.3, backtrack=True):
        super().__init__()
        if not 0 < prior_std_min < prior_std_max or noise_std <= 0:
            raise ValueError('Invalid prior/noise scale')
        self.embedding_dim = embedding_dim
        self.prior_std_min, self.prior_std_max = prior_std_min, prior_std_max
        self.backtrack = backtrack
        self.prior = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.SiLU(), nn.Linear(hidden, 10))
        # Fixed observation log-noise for this prototype. Tune/calibrate only on
        # training/validation data; callers may supply independently known SDs.
        self.register_buffer('noise_std', torch.tensor(float(noise_std)))

    def molecular_prior(self, embedding):
        if embedding.shape != (self.embedding_dim,) or not torch.isfinite(embedding).all():
            raise ValueError('Expected one finite molecular vector')
        raw = self.prior(embedding)
        mu = raw[:5]
        std = self.prior_std_min + (self.prior_std_max - self.prior_std_min) * torch.sigmoid(raw[5:])
        return mu, std

    @staticmethod
    def log_curve(log_parameters, events, times):
        parameters = log_parameters.exp()
        if not torch.isfinite(parameters).all():
            raise ValueError('Nonfinite positive PK parameters')
        curve = event_curves(parameters, events, times)
        if not torch.isfinite(curve).all() or (curve < -1e-8).any():
            raise ValueError('Invalid compartment concentration')
        return torch.log(curve.clamp_min(0) + 1e-5)

    def sensitivity(self, log_parameters, events, times, *, create_graph):
        # Works under torch.no_grad during inference. torch.inference_mode is
        # unsuitable because adaptation intrinsically needs local derivatives.
        with torch.enable_grad():
            x = log_parameters if create_graph else log_parameters.detach().requires_grad_(True)
            function = lambda value: self.log_curve(value, events, times.tolist())
            jacobian = torch.autograd.functional.jacobian(function, x, create_graph=create_graph)
        return jacobian if create_graph else jacobian.detach()

    def condition(self, embedding, events, support_times, support_log_c, *, support_noise_std=None):
        if support_times.ndim != 1 or support_times.shape != support_log_c.shape:
            raise ValueError('Support vectors must be aligned')
        if not torch.isfinite(support_times).all() or not torch.isfinite(support_log_c).all():
            raise ValueError('Nonfinite support')
        if not events:
            raise ValueError('Dose history is required')
        mu, prior_std = self.molecular_prior(embedding)
        graph = torch.is_grad_enabled()
        sigma = self.noise_std if support_noise_std is None else support_noise_std
        sigma = torch.as_tensor(sigma, dtype=mu.dtype, device=mu.device)
        if not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError('Positive finite observation SD required')
        if len(support_times):
            base = self.log_curve(mu, events, support_times.tolist())
            jacobian = self.sensitivity(mu, events, support_times, create_graph=graph)
        else:
            base, jacobian = mu.new_empty(0), mu.new_empty((0, 5))
        updated = gaussian_information_update(mu, prior_std, base, jacobian, support_log_c, sigma)
        alpha = 1. if len(support_times) else 0.
        support_objective_before, support_objective_after = 0., 0.
        if len(support_times):
            def objective(x):
                errors = (self.log_curve(x, events, support_times.tolist()) - support_log_c) / sigma
                return .5 * (errors.square().sum() + ((x - mu) / prior_std).square().sum())
            with torch.no_grad():
                support_objective_before = float(objective(mu))
                shift = updated['mean'] - mu
                if self.backtrack:
                    # Include alpha=0, so this support MAP objective never grows.
                    candidates = [1., .5, .25, .125, 0.]
                    evaluated = []
                    for size in candidates:
                        try:
                            value = float(objective(mu + size * shift))
                        except (ValueError, RuntimeError):
                            value = math.inf
                        evaluated.append(value if math.isfinite(value) else math.inf)
                    index = min(range(len(candidates)), key=lambda i: (evaluated[i], -candidates[i]))
                    alpha = candidates[index]
                support_objective_after = float(objective(mu + alpha * shift))
            updated['mean'] = mu + alpha * (updated['mean'] - mu)
            updated['whitened_shift'] = alpha * updated['whitened_shift']
        updated.update(prior_mean=mu, prior_std=prior_std, support_jacobian=jacobian,
                       support_cutoff=float(support_times.max()) if len(support_times) else -math.inf,
                       step_size=alpha, support_objective_before=support_objective_before,
                       support_objective_after=support_objective_after,
                       covariance_scope='local Gaussian covariance at molecular prior; not calibrated nonlinear posterior')
        return updated

    def forecast(self, conditioned, events, query_times):
        if query_times.ndim != 1 or not len(query_times) or not torch.isfinite(query_times).all():
            raise ValueError('Expected finite nonempty queries')
        if (query_times <= conditioned['support_cutoff']).any():
            raise ValueError('Queries must be strictly after support')
        mean = conditioned['mean']
        prediction = self.log_curve(mean, events, query_times.tolist())
        j = self.sensitivity(mean, events, query_times, create_graph=torch.is_grad_enabled())
        latent_variance = (j @ conditioned['covariance'] * j).sum(-1).clamp_min(0)
        return dict(log_concentration=prediction, latent_log_variance=latent_variance,
                    predictive_log_variance=latent_variance + self.noise_std.square(),
                    parameters=mean.exp(), conditioning=conditioned)

    def forward(self, embedding, events, support_times, support_log_c, query_times, *, support_noise_std=None):
        if len(support_times) and (query_times <= support_times.max()).any():
            raise ValueError('Queries must be strictly after support')
        state = self.condition(embedding, events, support_times, support_log_c, support_noise_std=support_noise_std)
        return self.forecast(state, events, query_times)


def marginal_gaussian_nll(prediction, query_log_c):
    """Per-query marginal Gaussian NLL; not a joint trajectory likelihood.

    Query labels enter only here, outside conditioning. Episode weighting and
    outer split isolation are the responsibility of the training procedure.
    """
    mean, variance = prediction['log_concentration'], prediction['predictive_log_variance']
    if query_log_c.shape != mean.shape or not torch.isfinite(query_log_c).all():
        raise ValueError('Query targets must match finite predictions')
    return .5 * (math.log(2 * math.pi) + variance.log() + (query_log_c - mean).square() / variance).mean()
