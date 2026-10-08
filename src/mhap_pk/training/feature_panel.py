"""Reuse the exact frozen legacy training function with an isolated feature provider.

Function-local global bindings avoid editing historical source or process-global
monkeypatching. No equations, losses, optimizer or evaluation code are copied.
"""
from types import FunctionType
from .traceable_panel import run as legacy_run
from ..data.molecular_encoders import representations

def run(fold,seed,output_root,representation,partition_spec=None):
    def features(meta,indices,unused_name,unused_seed):
        return representations(meta,indices,representation)
    bound=FunctionType(legacy_run.__code__,dict(legacy_run.__globals__,representations=features),legacy_run.__name__,legacy_run.__defaults__,legacy_run.__closure__)
    return bound(fold,seed,'molecular',output_root=output_root,partition_spec=partition_spec)
