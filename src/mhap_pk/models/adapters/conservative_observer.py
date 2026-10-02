"""Controlled innovation geometry; keeps dose dynamics and frozen prior unchanged."""
import math
import torch
from ...relational_model import RelationalPK

class ConservativeObserver(RelationalPK):
    def __init__(self, prior_state, initial_gain=.05, geometry='bounded_log', parameter=True):
        if not 0 < initial_gain < 1 or geometry not in ('amount','bounded_log'):
            raise ValueError('Invalid gain or geometry')
        super().__init__(parameter_update=parameter)
        self.double()
        self.prior.load_state_dict(prior_state)
        for p in self.prior.parameters(): p.requires_grad_(False)
        if not parameter:
            for p in self.parameter_head.parameters(): p.requires_grad_(False)
        self.geometry=geometry
        with torch.no_grad(): self.gain_head.bias.fill_(math.log(initial_gain/(1-initial_gain)))

    def condition(self,*args,**kwargs):
        kwargs['state_correction']=self.geometry
        kwargs['log_update_bound']=math.log(2.)
        return super().condition(*args,**kwargs)
