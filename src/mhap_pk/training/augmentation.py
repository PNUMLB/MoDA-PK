"""Training-only support views and perturbations of a fixed PK prior."""
import numpy as np
import torch


def support_view(times, labels, mask, generator, drop_probability=0., noise_sd=0.):
    if not 0 <= drop_probability <= 1 or noise_sd < 0:
        raise ValueError('Invalid augmentation magnitude')
    out_mask = mask.clone(); out_labels = labels.clone()
    for row in range(len(times)):
        blocks = torch.unique(times[row, mask[row]], sorted=True)
        for time in blocks:
            same = mask[row] & (times[row] == time)
            if drop_probability and time != blocks[-1]:
                if torch.rand((), generator=generator) < drop_probability:
                    out_mask[row, same] = False
            if noise_sd:
                # One perturbation per time block preserves replicate differences.
                noise = torch.randn((), dtype=labels.dtype, generator=generator).clamp(-2., 2.) * noise_sd
                out_labels[row, same] += noise
    return out_labels, out_mask


def training_noise_scale(profiles, fitting):
    squared = 0.; df = 0; blocks = 0
    for index in fitting:
        support = profiles[int(index)]['episodes'][-1].support
        for _, group in support.groupby('time'):
            values = group.target_log_dv.to_numpy()
            if len(values) > 1:
                squared += float(((values-values.mean())**2).sum())
                df += len(values)-1; blocks += 1
    raw = float(np.sqrt(squared/df)) if df else None
    # This fallback is a registered robustness stress, not measured assay noise.
    estimated = raw is not None and raw > 0
    sd = float(np.clip(raw, .01, .10)) if estimated else .05
    return dict(noise_sd=sd,replicate_sd=raw,replicate_df=df,replicate_blocks=blocks,
                origin='training_support_replicates' if estimated else 'predeclared_synthetic_stress_0.05')


def perturb_prior(mu, supported, scales, generator, probability=.5):
    if not 0 <= probability <= 1:
        raise ValueError('Invalid probability')
    keep = (torch.rand(len(mu), generator=generator) < probability) & supported
    eps = torch.randn(len(mu), 2, dtype=mu.dtype, generator=generator).clamp(-2., 2.) * scales
    shift = torch.zeros_like(mu)
    shift[:, 1:3] = eps * keep[:, None]
    return mu + shift, shift
