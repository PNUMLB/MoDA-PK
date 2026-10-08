"""Replay adopted primary checkpoints and diagnose prior/parameter/state contributions."""
from pathlib import Path
import sys,json,hashlib
import torch,pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src')]
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch
from mhap_pk.training.run_io import inputs
from mhap_pk.models.priors.molecular_identity import build_joint_input
from mhap_pk.relational import floor_curve,floor_concentration
from mhap_pk.evaluation.drug_metrics import balanced_metrics
S=R/'results/summaries/hybrid_primary_20261002';ps,meta,_=load_profiles();torch.set_num_threads(1)
refs=pd.read_parquet(S/'primary_predictions.parquet');refs=refs[(refs.condition=='ours')&(refs.k==2)&(refs.protocol=='first_k')];rows=[];traces=[];checks=[]
for (fold,seed),ref in refs.groupby(['fold','seed']):
 root=R/ref.run_id.iloc[0];tr,va,te=indices_for_split(meta,fold);z,ids,sc=representations(meta,tr,'unimol',17);b=make_batch(ps,meta,te,z,ids);pop=torch.tensor(json.loads((root/'population.json').read_text())['log_parameters'],dtype=torch.double);m=build_joint_input('molecular',len(sc['fitted_drugs']),pop);m.load_state_dict(torch.load(root/'selected.pt',weights_only=True));m.eval();args=inputs(b,2)
 with torch.no_grad():
  mu=m.prior(b.z,b.ids)[0];out=m(*args);ys={'full':out['mean'],'prior':floor_curve(mu,b.events,b.query_t)}
  m.state_update=False;ys['parameter_path']=m(*args)['mean'];m.state_update=True;m.parameter_update=False;ys['state_path']=m(*args)['mean']
  for step in out['state']['trace']:
   post=floor_concentration(step['parameters'],step['state'])
   for i in torch.where(step['active'])[0].tolist():
    p=ps[te[i]];t=float(step['time'][i]);s=p['episodes'][2].support;actual=float(s[s.time==t].target_log_dv.mean());traces.append(dict(fold=fold,seed=seed,**{k:p[k] for k in ['source_id','drug_id','participant_id','profile_id']},time=t,observed_log=actual,before_update=float(step['log_prediction'][i]),after_update=float(post[i]),gain=float(step['gain'][i]),cl_offset=float(step['parameters'][i,1]-mu[i,1]),vc_offset=float(step['parameters'][i,2]-mu[i,2])))
  for mode,y in ys.items():
   for i,idx in enumerate(te):
    p=ps[idx]
    for j in torch.where(b.query_mask[i])[0].tolist():rows.append(dict(fold=fold,seed=seed,mode=mode,**{k:p[k] for k in ['source_id','drug_id','participant_id','profile_id']},observation_id=str(p['query'].observation_id.iloc[j]),query_time=float(b.query_t[i,j]),y_true_log=float(b.query_y[i,j]),y_pred_log=float(y[i,j])))
  actual=np.concatenate([ys['full'][i,b.query_mask[i]].numpy() for i in range(len(te))]);want=ref.set_index('observation_id').loc[[str(q) for idx in te for q in ps[idx]['query'].observation_id]].y_pred_log.to_numpy();err=float(abs(actual-want).max());assert err<1e-12;checks.append(dict(fold=fold,seed=seed,max_difference=err))
p=pd.DataFrame(rows);p['track']=np.where(p.fold=='known','Known-drug','Joint drug–source');p.to_parquet(S/'stage_predictions.parquet',index=False);pd.DataFrame(traces).to_csv(S/'support_update_traces_LOCAL_ONLY.csv',index=False);pd.DataFrame(checks).to_csv(S/'checkpoint_replay_checks.csv',index=False)
cols=['track','seed','mode','source_id','drug_id'];d=pd.DataFrame([dict(zip(cols,k),**balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id+'::'+g.participant_id)) for k,g in p.groupby(cols)]);d.to_csv(S/'stage_drug_metrics.csv',index=False)
x=p[p['mode']=='full'].copy();x['e']=x.y_pred_log-x.y_true_log;x['sq']=x.e**2;pr=x.groupby(['track','seed','source_id','drug_id','participant_id','profile_id']).agg(mse=('sq','mean'),bias=('e','mean'),queries=('e','size')).reset_index();pr['bias_squared']=pr.bias**2;pr['centered_mse']=np.maximum(0,pr.mse-pr.bias_squared);pr.to_csv(S/'profile_bias_shape.csv',index=False)
print('Adopted-model checkpoint parity:',len(checks),'max',max(v['max_difference'] for v in checks))
