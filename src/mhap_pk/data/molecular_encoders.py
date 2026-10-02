"""Frozen feature cache selection; all fitted statistics are partition-local."""
from pathlib import Path
import hashlib
import numpy as np,torch
from .core import ROOT,representations as original
from .frozen_fingerprints import representations as ecfp

def scale_cached(meta,fitting_indices,drugs,embeddings,representation):
    train=sorted(meta.iloc[fitting_indices].drug_id.unique());all_drugs=sorted(meta.drug_id.unique())
    vectors=dict(zip(drugs,embeddings));dim=embeddings.shape[1]
    if not 0<dim<=512 or not np.isfinite(embeddings).all():raise ValueError('Finite features of at most512 dimensions required')
    matrix=np.stack([vectors[d] for d in all_drugs]);fit=matrix[[all_drugs.index(d) for d in train]]
    mean=fit.mean(0);sd=fit.std(0).clip(.1);normalized=(matrix-mean)/sd
    # Lossless padding preserves the existing512-input backbone; no PCA/test fitting.
    normalized=np.pad(normalized,((0,0),(0,512-dim)))
    z=torch.tensor(normalized[[all_drugs.index(d) for d in meta.drug_id]],dtype=torch.double)
    ids=torch.tensor([train.index(d) if d in train else -1 for d in meta.drug_id])
    sc=dict(fitted_drugs=train,representation=representation,raw_dimension=dim,output_dimension=512,padding='trailing zeros after train-fold normalization',mean=mean.tolist(),sd=sd.tolist(),minimum_feature_sd=.1)
    return z,ids,sc

def representations(meta,fitting_indices,name):
    if name=='unimol':return original(meta,fitting_indices,'unimol',17)
    if name=='ecfp':return ecfp(meta,fitting_indices)
    if name not in ['chemberta_mlm','chemberta_mtr']:raise ValueError(name)
    file=ROOT/'data/features/molecular_comparison_20261002'/f'{name}.npz'
    with np.load(file) as f:z,ids,sc=scale_cached(meta,fitting_indices,f['drugs'].astype(str),f['embeddings'].astype(np.float64),name)
    sc['cache_path']=str(file.relative_to(ROOT));sc['cache_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    return z,ids,sc
