from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.quality_metrics import extended_metrics
from mhap_pk.evaluation.drug_metrics import balanced_metrics
B=R/'results/runs/COMPLETION_SUITE/20261002';O=R/'results/summaries/completion_suite_20261002'
base=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet');base=base[(base.condition=='molecular_relation')&(base.protocol=='first_k')&(base.k==2)].copy();base['model']='gru'
frames=[base]
for variant in ['lstm','last_mlp','deep_sets','attention','synthetic','residual_centered']:
 for p in sorted((B/variant).glob('*/predictions.parquet')):
  m=json.loads((p.parent/'run_manifest.json').read_text());assert m['status']=='completed';assert hashlib.sha256(p.read_bytes()).hexdigest()==m['prediction_sha256']
  d=pd.read_parquet(p).rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});d=d[(d.protocol=='first_k')&(d.k==2)].copy();d['model']=variant;d['track']=np.where(d.fold=='known','Known-drug','Joint drug–source');frames.append(d)
d=pd.concat(frames,ignore_index=True);keys=['fold','seed','source_id','drug_id','participant_id','profile_id','observation_id']
assert not d.duplicated(['model']+keys).any()
ref=base.set_index(keys)[['query_time','y_true_log']].sort_index()
for m,g in d.groupby('model'):pd.testing.assert_frame_equal(ref,g.set_index(keys)[['query_time','y_true_log']].sort_index())
rows=[];drug=[]
for key,g in d.groupby(['track','model','seed']):rows.append(dict(zip(['track','model','seed'],key))|extended_metrics(g))
for key,g in d.groupby(['track','model','seed','drug_id']):drug.append(dict(zip(['track','model','seed','drug_id'],key))|balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id+'::'+g.participant_id))
c=pd.DataFrame(rows);v=pd.DataFrame(drug);c.to_csv(O/'controls_k2_by_seed.csv',index=False);v.to_csv(O/'controls_k2_drug_by_seed.csv',index=False)
d['error2']=(d.y_pred_log-d.y_true_log)**2
pk=['track','seed','source_id','drug_id','participant_id','profile_id']
p=d.groupby(['model']+pk).error2.mean().unstack('model');z=p.sub(p.gru,axis=0).drop(columns='gru');z.reset_index().to_csv(O/'controls_paired_profile_MSE_LOCAL_ONLY.csv',index=False)
a=[]
for key,g in z.groupby(level=['track','seed','drug_id']):
 for m in z.columns:a.append(dict(zip(['track','seed','drug_id'],key))|dict(model=m,profiles=len(g),harm_fraction=float((g[m]>1e-10).mean()),mean_MSE_difference=g[m].mean()))
pd.DataFrame(a).to_csv(O/'controls_harm_by_drug_seed.csv',index=False)
print(c.groupby(['track','model']).sb_hrmse.agg(['mean','std']).to_string())
print('Drug improvement counts vs GRU:')
x=v.groupby(['track','model','drug_id']).log_rmse.mean().unstack('model')
print(x.drop(columns='gru').lt(x.gru,axis=0).groupby('track').sum().to_string())
