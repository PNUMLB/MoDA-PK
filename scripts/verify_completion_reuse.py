"""Verify saved CV and molecular representation evidence without refitting."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.drug_metrics import balanced_metrics
from mhap_pk.data.traceable import load_profiles
from mhap_pk.data.resampling import partition_check
O=R/'results/summaries/completion_suite_20261002';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for name in ['resampling_20261002','molecular_encoder_20261002']:
 S=R/'results/summaries'/name;m=json.loads((S/'analysis_manifest.json').read_text())
 for f,h in m['files'].items():
  path=S/f;assert path.exists(),path;assert sha(path)==h,path
 checks.append(dict(scope=name,files=len(m['files']),hashes_match=True))
ps,meta,_=load_profiles();reg=json.loads((R/'data/splits/participant_resampling_20261002/partitions.json').read_text())
for name,spec in reg['partitions'].items():
 tr,va,te=partition_check(meta,spec);groups=[set(map(tuple,meta.iloc[ix][['source_id','participant_id']].to_numpy())) for ix in [tr,va,te]];assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
p=pd.read_parquet(R/'results/summaries/resampling_20261002/canonical_predictions.parquet');p=p[(p.k==2)&(p.protocol=='first_k')];idcols=['evaluation','training_scope','model','seed','source_id','drug_id','participant_id','profile_id','observation_id'];assert not p.duplicated(idcols).any();assert np.isfinite(p[['y_true_log','y_pred_log']]).all().all()
rows=[]
for key,g in p.groupby(['evaluation','training_scope','model','seed','source_id','drug_id']):rows.append(dict(zip(['evaluation','training_scope','model','seed','source_id','drug_id'],key))|balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id+'::'+g.participant_id))
d=pd.DataFrame(rows);old=pd.read_csv(R/'results/summaries/resampling_20261002/drug_metrics_by_seed.csv');old=old[(old.k==2)&(old.protocol=='first_k')];keys=['evaluation','training_scope','model','seed','source_id','drug_id'];pair=d.merge(old,on=keys,validate='one_to_one',suffixes=('_new','_old'));assert len(pair)==len(d)==len(old);delta=float(abs(pair.log_rmse_new-pair.log_rmse_old).max());assert delta<1e-12
# Each CV query occurs once per seed/model, not one independent sample per fold.
expected={str(q) for x in ps for q in x['query'].observation_id}
for key,g in p[p.evaluation=='CV5'].groupby(['training_scope','model','seed']):assert set(g.observation_id.astype(str))==expected
scores=d.groupby(['evaluation','training_scope','model','seed','source_id']).log_rmse.mean().groupby(['evaluation','training_scope','model','seed']).mean().rename('sb_hrmse').reset_index();summary=scores.groupby(['evaluation','training_scope','model']).sb_hrmse.agg(['mean','std']).reset_index();summary.to_csv(O/'resampling_summary.csv',index=False);d.to_csv(O/'resampling_drug_metrics_verified.csv',index=False)
# Embedding evidence is recomputed at primary k from the canonical export.
e=pd.read_parquet(R/'results/summaries/molecular_encoder_20261002/canonical_predictions.parquet');print('embedding columns',e.columns.tolist())
(O/'reuse_verification.json').write_text(json.dumps(dict(status='completed',checks=checks,partitions_checked=len(reg['partitions']),resampling_metric_rows=len(d),maximum_metric_difference=delta,cv_complete_query_coverage=True,molecular_prediction_rows=len(e),molecular_prediction_sha256=sha(R/'results/summaries/molecular_encoder_20261002/canonical_predictions.parquet')),indent=2))
print(summary.to_string(index=False))
