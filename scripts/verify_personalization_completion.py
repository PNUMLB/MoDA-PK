"""Independent checkpoint replay of primary predictions, without model selection."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles
from mhap_pk.data import representations,make_batch
from mhap_pk.training.run_io import inputs,digest,save_json
from mhap_pk.relational_model import RelationalPK
from mhap_pk.models.adapters.shared_prior import shared_observer
from mhap_pk.models.adapters.conservative_observer import ConservativeObserver
from mhap_pk.models.adapters.validated_map import fit_map,predict_map
root=Path(sys.argv[1]);torch.set_num_threads(1)
assert json.loads((root/'status.json').read_text())['status']=='completed'
for name,h in json.loads((root/'input_hashes.json').read_text()).items():assert digest(R/name)==h,name
plan=json.loads((root/'plan.json').read_text());part=plan['partitions']['original'];ps,meta,_=load_profiles();z,ids,_=representations(meta,np.array(part['train']),'unimol',17);b=make_batch(ps,meta,np.array(part['assessment']),z,ids);largest=0.;count=0
for seed in range(1,6):
 d=root/f'original__s{seed}';locked=json.loads((d/'selection_locked.json').read_text());pred=pd.read_parquet(d/'predictions.parquet');pred=pred[(pred.role=='assessment')&(pred.protocol=='first_k')&(pred.k==2)]
 base=RelationalPK(prior_only=True).double();base.load_state_dict(torch.load(d/'prior.pt',weights_only=True));prior=base.prior.state_dict()
 for name in pred.condition.unique():
  if name.startswith('map_'):
   result=fit_map(base.prior,inputs(b,2),name[4:],locked['map_scales'][name[4:]]);y=predict_map(result,b.events,b.query_t)
  else:
   if name in ['prior','prior_budget120']:m=RelationalPK(prior_only=True).double()
   elif name=='fixed_state':m=ConservativeObserver(prior,locked['fixed_gain'],'amount',False)
   else:m=shared_observer(prior,name)
   if name!='fixed_state':m.load_state_dict(torch.load(d/f'{name}.pt',weights_only=True))
   m.eval()
   with torch.no_grad():y=m(*inputs(b,2))['mean']
  expected=pred[pred.condition==name].set_index(['source_id','drug_id','participant_id','profile_id','query_id'])
  n=0
  for i,idx in enumerate(b.indices):
   p=ps[idx];key=tuple(p[c] for c in ['source_id','drug_id','participant_id','profile_id'])
   for j in torch.where(b.query_mask[i])[0].tolist():
    ref=float(expected.loc[key+(str(p['query'].observation_id.iloc[j]),)].predicted_log);err=abs(float(y[i,j].detach())-ref);largest=max(largest,err);n+=1
  assert n==len(expected);count+=n
assert largest<1e-9,largest
save_json(root/'independent_replay_verification.json',dict(status='passed',prediction_values_checked=count,max_absolute_difference=largest,conditions=9,seeds=5,scope='Original Known assessment k2, checkpoint reload and support-only MAP rerun; no selection',input_hashes_verified=True,verifier_hash=digest(Path(__file__))))
print(count,largest)
