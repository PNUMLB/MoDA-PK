"""Same-checkpoint update diagnostics, distinct from retrained ablation."""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.data import representations,make_batch
from mhap_pk.training.run_io import inputs
from mhap_pk.baselines.low_data import build
from mhap_pk.relational import floor_curve,floor_concentration
from mhap_pk.evaluation.drug_metrics import balanced_metrics
O=R/'results/summaries/completion_suite_20261002';O.mkdir(exist_ok=True);torch.set_num_threads(1)
p=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet');p=p[(p.k==2)&(p.protocol=='first_k')];refs=p[p.condition=='molecular_relation'];ps,meta,_=load_profiles();rows=[];traces=[];checks=[];preds=[]
for (fold,seed),ref in refs.groupby(['fold','seed']):
 rid=ref.run_id.iloc[0];root=R/rid if rid.startswith('results/') else R/'results/runs/TRACEABLE/evaluation_20260925'/rid
 tr,va,te=indices_for_split(meta,fold);z,ids,sc=representations(meta,tr,'unimol',17);b=make_batch(ps,meta,te,z,ids);pop=torch.tensor(json.loads((root/'population.json').read_text())['log_parameters'],dtype=torch.double);m=build('molecular',len(sc['fitted_drugs']),pop);m.load_state_dict(torch.load(root/'selected.pt',weights_only=True));m.eval();args=inputs(b,2)
 with torch.no_grad():
  mu=m.prior(b.z,b.ids)[0];out=m(*args);ys={'full':out['mean'],'same_checkpoint_prior':floor_curve(mu,b.events,b.query_t)}
  m.state_update=False;ys['parameter_path_only']=m(*args)['mean'];m.state_update=True;m.parameter_update=False;ys['state_path_only']=m(*args)['mean'];m.parameter_update=True
  for step in out['state']['trace']:
   post=floor_concentration(step['parameters'],step['state'])
   for i in torch.where(step['active'])[0].tolist():
    profile=ps[te[i]];t=float(step['time'][i]);s=profile['episodes'][2].support;actual=float(s.loc[s.time==t].target_log_dv.mean());traces.append(dict(fold=fold,seed=seed,**{k:profile[k] for k in ['source_id','drug_id','participant_id','profile_id']},time=t,observed_log=actual,before_update=float(step['log_prediction'][i]),after_update=float(post[i]),gain=float(step['gain'][i]),cl_offset=float(step['parameters'][i,1]-mu[i,1]),vc_offset=float(step['parameters'][i,2]-mu[i,2])))
  for mode,y in ys.items():
   for i,idx in enumerate(te):
    profile=ps[idx]
    for j in torch.where(b.query_mask[i])[0].tolist():preds.append(dict(fold=fold,seed=seed,mode=mode,**{k:profile[k] for k in ['source_id','drug_id','participant_id','profile_id']},observation_id=str(profile['query'].observation_id.iloc[j]),query_time=float(b.query_t[i,j]),y_true_log=float(b.query_y[i,j]),y_pred_log=float(y[i,j])))
  actual=np.concatenate([ys['full'][i,b.query_mask[i]].numpy() for i in range(len(te))]);wanted=ref.set_index('observation_id').loc[[str(q) for idx in te for q in ps[idx]['query'].observation_id]].y_pred_log.to_numpy();delta=float(abs(actual-wanted).max());assert delta<1e-12;checks.append(dict(fold=fold,seed=seed,max_difference=delta))
d=pd.DataFrame(preds);d.to_parquet(O/'stage_predictions_LOCAL_ONLY.parquet',index=False);pd.DataFrame(traces).to_csv(O/'stage_support_LOCAL_ONLY.csv',index=False);pd.DataFrame(checks).to_csv(O/'stage_replay_checks.csv',index=False)
for key,g in d.groupby(['fold','seed','mode','source_id','drug_id']):rows.append(dict(zip(['fold','seed','mode','source_id','drug_id'],key))|balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id+'::'+g.participant_id))
a=pd.DataFrame(rows);a.to_csv(O/'stage_drug_metrics.csv',index=False);a['track']=np.where(a.fold=='known','Known-drug','Joint drug–source');su=a.groupby(['track','mode','drug_id'])[['log_rmse','log_mae','signed_log_bias','within_twofold']].agg(['mean','std']);su.columns=['_'.join(x) for x in su.columns];su.reset_index().to_csv(O/'stage_summary.csv',index=False)
pd.DataFrame(traces).groupby(['fold','drug_id'])[['before_update','after_update','observed_log','gain','cl_offset','vc_offset']].mean().to_csv(O/'stage_support_summary.csv')
print('COMPLETE',len(checks),'checkpoint replays')
