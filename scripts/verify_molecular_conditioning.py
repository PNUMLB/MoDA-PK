"""Verify the matched-input contrast before interpreting paired effects."""
import json
import pandas as pd
from analyze_molecular_conditioning import parent,O,N,sha,R
checks=[]
for fold in ['known']+[f'loso_{i:02d}' for i in range(9)]:
 for seed in range(1,11):
  ref=None;cfgref=None;artref=None
  for mode in N:
   p=parent(fold,seed,mode);d=pd.read_parquet(p/'predictions.parquet');d=d[(d.protocol=='first_k')&(d.k==2)]
   ix=['source_id','drug_id','participant_id','profile_id','observation_id'];assert not d.duplicated(ix).any()
   a=d.set_index(ix)[['query_time','y_true_lnc']].sort_index();c=json.loads((p/'config.resolved.json').read_text());ar=json.loads((p/'artifacts.json').read_text())
   c={k:c[k] for k in ['dataset_id','split_sha256','epochs','learning_rate','weight_decay','validate_every']};ar={k:ar[k] for k in ['train','validation','test','scaler']}
   if ref is None:ref=a;cfgref=c;artref=ar
   else:assert ref.equals(a) and cfgref==c and artref==ar,(fold,seed,mode)
  checks.append(dict(fold=fold,seed=seed,queries=len(ref),models=len(N),exact_queries_labels_and_training_recipe=True))
pd.DataFrame(checks).to_csv(O/'matched_contract_checks.csv',index=False)
m=json.loads((O/'manifest.json').read_text());m['matched_contract_checks']=len(checks);m['source_sha256']=sha(R/'scripts/analyze_molecular_conditioning.py');m['verification_source_sha256']=sha(R/'scripts/verify_molecular_conditioning.py');(O/'manifest.json').write_text(json.dumps(m,indent=2));print('PASS',len(checks),'matched fold/seed comparisons')
