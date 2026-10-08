"""Verify training-only centering against the exact saved original random draws."""
from pathlib import Path
import sys,json,hashlib
import torch,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import make_batch
from mhap_pk.relational import floor_curve
B=R/'results/runs/COMPLETION_SUITE/20261002';O=R/'results/summaries/completion_suite_20261002';torch.set_num_threads(1);ps,meta,_=load_profiles();rows=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for fold in json.loads((B/'residual_centered/plan.json').read_text())['folds']:
 tr,_,_=indices_for_split(meta,fold)
 b=make_batch(ps,meta,tr,torch.zeros(len(meta),512,dtype=torch.double),torch.zeros(len(meta),dtype=torch.long))
 for seed in range(1,11):
  name=f'{fold}__s{seed}__molecular_relation';p=B/'residual_centered'/name;q=B/'synthetic'/name;info=json.loads((p/'synthetic_generator.json').read_text());assert info['fitting_indices']==tr.tolist();assert info['original_trajectory_sha256']==sha(q/'synthetic_training_LOCAL_ONLY.pt');assert info['original_generator_sha256']==sha(q/'synthetic_generator.json')
  original=torch.load(q/'synthetic_training_LOCAL_ONLY.pt',weights_only=True);centered=torch.load(p/'synthetic_training_LOCAL_ONLY.pt',weights_only=True)
  with torch.no_grad():h0=floor_curve(original['teacher_parameters'],b.events,b.support_t[:,3]);q0=floor_curve(original['teacher_parameters'],b.events,b.query_t)
  hd=(centered['support']-(b.support_y[:,3]+original['support']-h0)).abs().max().item();qd=(centered['query']-(b.query_y+original['query']-q0)).abs().max().item();assert max(hd,qd)<1e-12
  rows.append(dict(fold=fold,seed=seed,same_random_draws=True,train_partition_exact=True,support_max_delta=hd,query_max_delta=qd,source_sha256=info['original_trajectory_sha256']))
pd.DataFrame(rows).to_csv(O/'centered_augmentation_verification.csv',index=False)
print('Verified',len(rows),'training-only centered augmentation datasets')
