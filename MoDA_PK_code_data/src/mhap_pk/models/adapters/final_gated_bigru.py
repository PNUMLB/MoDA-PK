"""User-adopted MoDE-PK: Uni-Mol + identity + causal-prefix BiGRU + gate."""
import torch
from .gated_encoder_panel import build_gated_encoder
from .encoder_controls import HistoryCell
from ..priors.hybrid_controls import BranchControlPrior

def build_final(variant,n,pop):
 m=build_gated_encoder('bigru',n,pop)
 if variant=='without_shared_context':m.share_observer_context=False
 if variant=='without_gate':del m.dose_controller;m.dosing_kind='none'
 if variant in ['without_updates','parameter_only','state_only']:
  m.parameter_update=variant=='parameter_only';m.state_update=variant=='state_only'
 if variant in ['without_molecule','without_identity']:
  with torch.random.fork_rng(devices=[]):prior=BranchControlPrior(n).double()
  prior.load_state_dict(m.prior.state_dict());prior.drop_molecule=variant=='without_molecule';prior.drop_identity=variant=='without_identity';m.prior=prior
 if variant=='memoryless':m.gru=HistoryCell('last_mlp').double()
 return m
