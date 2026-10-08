"""Independently trained observer components around one immutable prior."""
from ...relational_model import RelationalPK

def shared_observer(prior_state,mode):
    flags={'parameter':(True,False),'state':(False,True),'combined':(True,True)}
    if mode not in flags:raise ValueError(mode)
    parameter,state=flags[mode]
    model=RelationalPK(parameter_update=parameter,state_update=state).double()
    model.prior.load_state_dict(prior_state)
    for p in model.prior.parameters():p.requires_grad_(False)
    if not parameter:
        for p in model.parameter_head.parameters():p.requires_grad_(False)
    if not state:
        for p in model.gain_head.parameters():p.requires_grad_(False)
    return model
