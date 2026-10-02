from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles
from mhap_pk.data.core import indices_for_split
from mhap_pk.training.run_io import save_json,digest
S=R/'data/splits/source_traceable_20260925';assert not S.exists();S.mkdir(parents=True)
ps,meta,ex=load_profiles();assert len(meta)==482 and meta.drug_id.nunique()==11 and meta.source_id.nunique()==9
folds={};audit=[]
for f in ['known']+[f'loso_{i:02d}' for i in range(9)]:
 tr,va,te=indices_for_split(meta,f);folds[f]=dict(train=tr.tolist(),validation=va.tolist(),test=te.tolist())
 for role,ix in [('train',tr),('validation',va),('test',te)]:
  for j in ix:audit.append(dict(fold=f,role=role,index=int(j),**meta.iloc[j].to_dict()))
 if f=='known':assert set(meta.iloc[tr].drug_id)==set(meta.iloc[va].drug_id)==set(meta.iloc[te].drug_id)
 else:assert set(meta.iloc[tr].drug_id).isdisjoint(meta.iloc[te].drug_id)
spec=dict(dataset_id='source_traceable_20260925',split_id='source_traceable_20260925',profile_keys=meta[['source_id','drug_id','participant_id','profile_id']].astype(str).values.tolist(),folds=folds,source_order=sorted(meta.source_id.unique()),policy='Known participant assignments preserved; Joint source ring regenerated over nine retained legacy source groups; not independent original-study validation',membership_sha256=digest(R/'data/metadata/provenance_groups_20260925/source_traceable/dataset_reference.json'))
save_json(S/'partitions.json',spec);pd.DataFrame(audit).to_csv(S/'membership.csv',index=False);ex.to_csv(S/'loader_exclusions.csv',index=False)
cfg=dict(experiment='TRACEABLE_20260925',dataset_id=spec['dataset_id'],split_id=spec['split_id'],split_sha256=digest(S/'partitions.json'),folds=list(folds),seeds=[1,2,3,4,5],modes=['molecular','molecular_relation','no_molecule','drug_id','ecfp','chemical_only','pk_only','molecular_offset','offset_relation','dcm','deepct','unipk'],epochs=120,learning_rate=.003,weight_decay=.0001,validate_every=10,fit_timeout_seconds=7200,workers=20,population_learning_rate=.01,map_steps=50,protocols=['first_k','common_origin'],k=[0,1,2,4],selection='Frozen hyperparameters and held-out validation checkpoint selection; no test-driven tuning',scope='11 source-traceable drugs; retained data still have clinical QC gaps. Fresh training, no old learned checkpoints/calibration. Existing frozen molecular caches are allowed.',future_doses='Recorded ledger convention retained; known-at-cutoff remains unverified where absent',poppk_scope='Drug-specific VI+MAP Known only; unavailable Joint',source_manifest_sha256=digest(R/'data/metadata/provenance_groups_20260925/manifest.json'))
save_json(R/'configs/experiments/source_traceable_20260925.json',cfg);print('REGISTERED',len(folds)*5*len(cfg['modes']),'neural fits, 5 Known PopPK bundles')
