"""Fill k=3 inference missing from the reused seed-1 240epoch budget experiment."""
from pathlib import Path
import sys,json,time,hashlib
import torch,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch
from mhap_pk.evaluation.history_protocols import initial_observation_inputs
from complete_comparison_refresh import build
OLD=R/'results/runs/COMPLETION_SUITE/20261002/budget_seed1_k3';checks=[]
B=R/'results/runs/COMPLETION_SUITE/20261002/budget_seed1_k3_verified';B.mkdir(exist_ok=True);torch.set_num_threads(1);ps,meta,_=load_profiles();plan=json.loads((R/'results/runs/COMPARISON_REFRESH/20260929/budget240/plan.json').read_text())
for fold in plan['folds']:
 tr,va,te=indices_for_split(meta,fold);z,ids,sc=representations(meta,tr,'unimol',17);b=make_batch(ps,meta,te,z,ids)
 for mode in ['molecular_relation','dcm','unipk','deepct']:
  out=B/f'{fold}__{mode}.parquet'
  if out.exists():continue
  p=R/f'results/runs/COMPARISON_REFRESH/20260929/budget240/{fold}__s1__{mode}';pop=torch.tensor(json.loads((p/'population.json').read_text())['log_parameters'],dtype=torch.double);m=build('molecular' if mode=='molecular_relation' else mode,len(sc['fitted_drugs']),pop);m.load_state_dict(torch.load(p/'selected.pt',weights_only=True));m.eval()
  args=initial_observation_inputs(b,3)
  with torch.no_grad():y=m(*args)['mean']
  rows=[]
  for i,idx in enumerate(te):
   profile=ps[idx]
   for j in torch.where(b.query_mask[i])[0].tolist():rows.append(dict(fold=fold,seed=1,condition=mode,k=3,protocol='first_k',**{k:profile[k] for k in ['source_id','drug_id','participant_id','profile_id']},observation_id=str(profile['query'].observation_id.iloc[j]),query_time=float(b.query_t[i,j]),support_end_time=float(args[2][i,args[4][i]].max()),cutoff_time=float(args[2][i,args[4][i]].max()),fallback_flag=False,y_true_log=float(b.query_y[i,j]),y_pred_log=float(y[i,j])))
  new=pd.DataFrame(rows);old=pd.read_parquet(OLD/out.name);keys=['source_id','drug_id','participant_id','profile_id','observation_id'];aa=new.set_index(keys).sort_index();bb=old.set_index(keys).sort_index();pd.testing.assert_frame_equal(aa[sorted(aa.columns)],bb[sorted(bb.columns)],atol=1e-10,rtol=0);checks.append(dict(fold=fold,mode=mode,max_prediction_difference=float((aa.y_pred_log-bb.y_pred_log).abs().max()),queries=len(new),previous_sha256=hashlib.sha256((OLD/out.name).read_bytes()).hexdigest()));new.to_parquet(out,index=False);print(fold,mode,checks[-1]['max_prediction_difference'],flush=True)
(B/'manifest.json').write_text(json.dumps(dict(status='completed',fits=0,inferences=40,seed=1,k=3,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(B.glob('*.parquet'))}),indent=2))

pd.DataFrame(checks).to_csv(B/'replay_checks.csv',index=False)
