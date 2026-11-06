"""Controlled encoder/identity swaps with the tested dose-conditioned gate."""
import torch
from .dosing_conditioning import build_dosing
from .encoder_controls import HistoryCell
from .mamba_observer import MambaObserver
from ..priors.hybrid_controls import BranchControlPrior


def build_combination(encoder,identity,gate,n_drugs,population):
 if encoder not in ['gru','lstm','mamba']:raise ValueError(encoder)
 m=build_dosing('dose_gate',n_drugs,population)
 if not gate:
  del m.dose_controller
  m.dosing_kind='none'
 if not identity:
  with torch.random.fork_rng(devices=[]):prior=BranchControlPrior(n_drugs).double()
  prior.load_state_dict(m.prior.state_dict());prior.drop_molecule=False;prior.drop_identity=True;m.prior=prior
 if encoder=='lstm':m.gru=HistoryCell('lstm').double()
 elif encoder=='mamba':m.gru=MambaObserver().double()
 return m
