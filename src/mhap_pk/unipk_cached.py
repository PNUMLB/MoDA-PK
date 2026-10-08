"""Algebraically identical Uni-PK RHS with constant context products cached.

For fixed molecular context, W[z, C] + b == (W_z z + b) + w_C C.
The original parameters, gradients, solver, events and objective are preserved.
"""
import torch
import torch.nn.functional as F
from .literature_adapters import UniPKAdapter


class CachedUniPKAdapter(UniPKAdapter):
    def curve(self, z, events, times, rtol=1e-5, atol=1e-7):
        model = self.native.cmptmodel
        original = model.forward
        net, poka = model.net, model.poka
        net_context = F.linear(z, net[0].weight[:, :-1], net[0].bias)
        ka_context = F.linear(z, poka[0].weight[:, :-1], poka[0].bias)

        def rhs(t, y, params, volume):
            central = y[0, :, None]
            value = net[-1](net[2](net[1](net_context+central*net[0].weight[:, -1])))
            ka = poka[-1](poka[2](poka[1](ka_context+central*poka[0].weight[:, -1]))).squeeze(-1)
            rates = value[:, 1:].reshape(-1, 2, model.num_compartments-1)
            elimination = value[:, 0]/volume*y[0]
            flows = rates[:, 0].T*y[0]-rates[:, 1].T*y[1:3]
            absorption = ka*y[-1]
            return torch.cat([(-elimination-flows.sum(0)+absorption)[None], flows, (-absorption)[None]], 0)
        model.forward = rhs
        try:
            return super().curve(z, events, times, rtol=rtol, atol=atol)
        finally:
            model.forward = original
