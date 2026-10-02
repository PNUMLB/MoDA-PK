"""A deployable composition of the two frozen forecasting experts.

Routing uses only exact membership in the fitting molecular-vector registry.
It does not use task labels, drug names, support residuals or query outcomes.
This is a familiarity baseline, not calibrated uncertainty or a learned gate.
"""
import torch
from torch import nn
from .iterated_information import predict as information_predict


class FamiliarityRouter(nn.Module):
    def __init__(self, sequential, information, fitting_vectors, information_config):
        super().__init__()
        if fitting_vectors.ndim != 2 or fitting_vectors.shape[1] != 512 or not len(fitting_vectors):
            raise ValueError('A nonempty fitting-only [molecules,512] registry is required')
        if not torch.isfinite(fitting_vectors).all():
            raise ValueError('Registry vectors must be finite')
        self.sequential = sequential
        self.information = information
        self.register_buffer('fitting_vectors', fitting_vectors.detach().clone())
        self.information_config = dict(information_config)

    def familiarity(self, z):
        if z.ndim != 2 or z.shape[1] != 512 or not torch.isfinite(z).all():
            raise ValueError('Expected finite [batch,512] molecular vectors')
        # Direct differences avoid cancellation in the cdist matrix-product path.
        distance = (z[:, None]-self.fitting_vectors[None]).square().sum(-1).min(-1).values
        return distance <= 1e-20, distance

    def forward(self, z, events, support_t, support_y, support_mask, query_t,
                drug_indices=None, variance=False):
        if variance:
            raise ValueError('This composition only supplies point forecasts')
        seen, distance = self.familiarity(z)
        inputs = (z, events, support_t, support_y, support_mask, query_t, drug_indices)
        mean = torch.empty_like(query_t)
        for active, sequential in [(seen, True), (~seen, False)]:
            if not active.any(): continue
            selected = tuple(v[active] if isinstance(v, torch.Tensor) else v for v in inputs)
            if sequential:
                out = self.sequential(*selected, variance=False)
            else:
                out = information_predict(self.information, selected, self.information_config)
            mean[active] = out['mean']
        return dict(mean=mean, state=dict(seen_molecule=seen, registry_squared_distance=distance))
