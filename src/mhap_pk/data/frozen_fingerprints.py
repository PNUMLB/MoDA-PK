"""Frozen ECFP4 cache; normalization and drug index fitted only on training drugs."""
from pathlib import Path
import numpy as np,torch
from .core import ROOT

def representations(meta,fitting_indices):
 drugs=sorted(meta.drug_id.unique());training=sorted(meta.iloc[fitting_indices].drug_id.unique())
 with np.load(ROOT/'data/processed/full16_v1/embeddings/morgan.npz') as cache:vectors=dict(zip(cache['drugs'].astype(str),cache['embeddings'].astype(np.float64)))
 matrix=np.stack([vectors[d] for d in drugs]);assert matrix.shape[1]==512
 values=matrix[[drugs.index(d) for d in training]];mean=values.mean(0);sd=values.std(0).clip(.1);matrix=(matrix-mean)/sd
 z=torch.tensor(matrix[[drugs.index(d) for d in meta.drug_id]],dtype=torch.double);ids=torch.tensor([training.index(d) if d in training else -1 for d in meta.drug_id]);sc=dict(fitted_drugs=training,representation='ecfp4_512_nonchiral',mean=mean.tolist(),sd=sd.tolist(),minimum_feature_sd=.1)
 return z,ids,sc
