"""Canonical new-primary predictions and all baseline queries, independent of ablation jobs."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from mhap_pk.evaluation.quality_metrics import extended_metrics
from mhap_pk.evaluation.drug_metrics import balanced_metrics
from summarize_training_scope_extension import summarize
S=R/'results/summaries/hybrid_primary_20261002';S.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet';old=pd.read_parquet(base);baseline=old[old.condition.isin(['dcm','unipk','deepct','poppk_vi_map'])].copy();frames=[];inventory=[]
for path in (R/'results/runs/MOLECULAR_IDENTITY/20261002').glob('*__s*__molecular'):
 m=json.loads((path/'run_manifest.json').read_text());assert m['status']=='completed' and sha(path/'predictions.parquet')==m['prediction_sha256']
 p=pd.read_parquet(path/'predictions.parquet').rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});p['condition']='ours';p['track']='Known-drug' if m['case']=='known' else 'Joint drug–source';p['run_id']=str(path.relative_to(R));frames.append(p)
 inventory.append(dict(model='ours',run_id=str(path.relative_to(R)),prediction_sha256=sha(path/'predictions.parquet'),**json.loads((path/'selection.json').read_text())))
for rid,g in baseline.groupby('run_id'):
 path=R/rid if '/' in rid else R/('results/runs/TRACEABLE_POPPK/evaluation_20260925' if rid.startswith('full__s') else 'results/runs/TRACEABLE/evaluation_20260925')/rid
 m=json.loads((path/'run_manifest.json').read_text());assert m['status']=='completed' and sha(path/'predictions.parquet')==m['prediction_sha256']
 raw=pd.read_parquet(path/'predictions.parquet').rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});key=['seed','k','protocol','observation_id'];a=g.set_index(key);b=raw.set_index(key);common=a.index.intersection(b.index);assert len(common)>0
 assert np.array_equal(a.loc[common,['y_true_log','y_pred_log']],b.loc[common,['y_true_log','y_pred_log']]),rid
 inventory.append(dict(model=g.condition.iloc[0],run_id=str(path.relative_to(R)),prediction_sha256=sha(path/'predictions.parquet')))
frames.append(baseline);p=pd.concat(frames,ignore_index=True)
p['cutoff_time_recorded']=p.cutoff_time;p.loc[p.protocol=='first_k','cutoff_time']=p.loc[p.protocol=='first_k','support_end_time']
p['dataset_id']=p['split_id']='source_traceable_20260925'
assert np.isfinite(p[['y_true_log','y_pred_log']]).all().all();assert not p.duplicated(['condition','fold','seed','protocol','k','observation_id']).any()
keys=['source_id','drug_id','participant_id','profile_id','observation_id'];checks=[]
for (track,seed,protocol,k),g in p.groupby(['track','seed','protocol','k']):
 ref=g[g.condition=='ours'].set_index(keys)[['query_time','y_true_log','support_end_time']].sort_index()
 for model,h in g.groupby('condition'):
  pd.testing.assert_frame_equal(ref,h.set_index(keys)[ref.columns].sort_index());checks.append(dict(track=track,seed=seed,protocol=protocol,k=int(k),model=model,queries=len(h)))
p.to_parquet(S/'primary_predictions.parquet',index=False);pd.DataFrame(checks).to_csv(S/'primary_query_checks.csv',index=False);pd.DataFrame(inventory).to_csv(S/'primary_inventory.csv',index=False)
cols=['track','condition','seed','protocol','k'];c=pd.DataFrame([dict(zip(cols,key),**extended_metrics(g)) for key,g in p.groupby(cols)])
dcols=cols+['source_id','drug_id'];d=pd.DataFrame([dict(zip(dcols,key),**balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id+'::'+g.participant_id)) for key,g in p.groupby(dcols)])
fields=['log_rmse','log_mae','within_twofold','signed_log_bias','linear_rmse','linear_mae','r2_log','r2_linear'];ds=summarize(d,['track','condition','protocol','k','source_id','drug_id','participants','queries'],fields)
cs=summarize(c,['track','condition','protocol','k'],[x for x in c if x not in cols]);c.to_csv(S/'primary_cohort_by_seed.csv',index=False);cs.to_csv(S/'primary_cohort_summary.csv',index=False);d.to_csv(S/'primary_drug_by_seed.csv',index=False);ds.to_csv(S/'primary_drug_summary.csv',index=False)
# Descriptive clustered contrasts: bootstrap existing source effects, not independent seed patients.
x=d[(d.k==2)&(d.protocol=='first_k')];pair=[];boot=[]
for track in x.track.unique():
 for model in sorted(set(x.condition)-{'ours'}):
  a=x[(x.track==track)&(x.condition=='ours')].merge(x[(x.track==track)&(x.condition==model)],on=['seed','source_id','drug_id'],suffixes=('','_reference'))
  if a.empty:continue
  a['delta_log_rmse']=a.log_rmse-a.log_rmse_reference;a['reference']=model;pair.append(a)
  source=a.groupby(['source_id','drug_id']).delta_log_rmse.mean().groupby('source_id').mean();rng=np.random.default_rng(20261002);v=np.mean(rng.choice(source.to_numpy(),size=(5000,len(source)),replace=True),axis=1)
  boot.append(dict(track=track,reference=model,mean_delta=source.mean(),source_count=len(source),ci025=np.quantile(v,.025),ci975=np.quantile(v,.975),interpretation='Exploratory source bootstrap after seed averaging; conditional on fitted models and reused cohort'))
pd.concat(pair,ignore_index=True).to_csv(S/'primary_paired_drugs.csv',index=False);pd.DataFrame(boot).to_csv(S/'primary_source_bootstrap.csv',index=False)
# Forecast horizon bins fixed in hours, count rows and participants separately for every drug.
x=p[(p.k==2)&(p.protocol=='first_k')].copy();x['horizon']=x.query_time-x.support_end_time;assert (x.horizon>0).all();x['horizon_bin']=pd.cut(x.horizon,[0,1,4,12,24,np.inf],labels=['0–1','1–4','4–12','12–24','>24'])
hcols=['track','condition','seed','source_id','drug_id','horizon_bin'];h=pd.DataFrame([dict(zip(hcols,key),**balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id+'::'+g.participant_id)) for key,g in x.groupby(hcols,observed=True)]);h.to_csv(S/'horizon_by_seed.csv',index=False)
# Compare k0 and k2 on identical query identities, keeping same fits.
x=p[p.condition=='ours'].copy();x['sq']=(x.y_pred_log-x.y_true_log)**2;kcols=['track','seed','protocol','source_id','drug_id','participant_id','profile_id'];g=x.groupby(kcols+['k']).sq.mean().rename('mse').reset_index();effect=g[g.k==2].merge(g[g.k==0],on=kcols,suffixes=('','_prior'));effect['delta_mse']=effect.mse-effect.mse_prior;effect['worsened']=effect.delta_mse>1e-10;effect.to_csv(S/'history_effect_profiles.csv',index=False)
(S/'primary_manifest.json').write_text(json.dumps(dict(status='completed',new_primary='MolecularIdentityPrior + original GRU/PK, no relations',primary_fits=100,controls_unchanged=True,query_checks=len(checks),prediction_sha256=sha(S/'primary_predictions.parquet'),primary_input_root='results/runs/MOLECULAR_IDENTITY/20261002',baseline_input=str(base.relative_to(R)),baseline_input_sha256=sha(base),source_sha256=sha(Path(__file__))),indent=2))
print(cs[(cs.k==2)&(cs.protocol=='first_k')][['track','condition','sb_hrmse_mean','sb_hrmse_std']].to_string(index=False))
