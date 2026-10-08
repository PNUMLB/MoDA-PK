"""Essential E02: source replay, input contracts, primary-table and run provenance audit."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.data.traceable import load_profiles,indices_for_split
from mhap_pk.evaluation.drug_metrics import balanced_metrics
from mhap_pk.evaluation.quality_metrics import extended_metrics
S=R/'results/summaries/essential_20261002';S.mkdir(exist_ok=True);B=R/'results/runs/ESSENTIAL_20261002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str))
ps,meta,_=load_profiles();tr,va,te=indices_for_split(meta,'known')
# Recheck the public distribution against every canonical fluorouracil record.
raw=B/'raw_reference';c=pd.read_csv(raw/'fluorouracil_conc.csv');dem=pd.read_csv(raw/'fluorouracil_demog.csv');dose=pd.read_csv(raw/'fluorouracil_dose.csv');c=c.merge(dem[['ID','WEIGHT']],on='ID',validate='many_to_one');c['id_key']=c.ID.astype(int).astype(str);c['time_h']=c.TIME/60
original=pd.read_csv(R/'data/processed/full16_v1/primary_full16.csv',low_memory=False);g=original[original.dataset=='fluorouracil'].copy();g['id_key']=g.subject_id.astype(str);g['time_h']=g.time_original/60
joined=g.merge(c,on=['id_key','time_h'],validate='one_to_one');assert len(joined)==len(g)
missing=c.merge(g[['id_key','time_h']],on=['id_key','time_h'],how='left',indicator=True).query("_merge=='left_only'");assert (missing.CONC==0).all();missing.to_csv(S/'S5_raw_rows_not_retained_LOCAL_ONLY.csv',index=False)
checks={'raw_concentration':float(abs(joined.target_dv_original-joined.CONC).max()),'time_h':float(abs(joined.time-joined.TIME/60).max()),'weight':float(abs(joined.wt-joined.WEIGHT).max()),'concentration_umol':float(abs(joined.target_dv-joined.CONC*1000/joined.molecular_weight).max()),'dose_umol':float(abs(joined.dose-joined.NDOSE*joined.WEIGHT*1000/joined.molecular_weight).max())};assert max(checks.values())<1e-9
role={i:k for k,v in [('train',tr),('validation',va),('assessment',te)] for i in v};s5records=[];episodes=[]
ledger=pd.read_csv(R/'data/processed/full16_v1/dose_events.csv')
for i,p in enumerate(ps):
 e=p['episodes'][2];h=e.support;q=p['query'];assert h.time.max()<q.time.min();assert set(h.observation_id).isdisjoint(q.observation_id)
 assert len(h.time.unique())==2 and q.time.min()>np.sort(pd.concat([h,q]).time.unique())[1]
 if p['drug_id']=='fluorouracil':
  for kind,f in [('support',h),('query',q)]:
   for row in f.itertuples():s5records.append(dict(role=role[i],use=kind,source_id=p['source_id'],participant_id=p['participant_id'],profile_id=p['profile_id'],observation_id=row.observation_id,time_h=row.time,concentration_umol=float(np.exp(row.target_log_dv)),log_concentration=row.target_log_dv))
  ds=dose[dose.ID.astype(int).astype(str)==p['profile_id'].split('::')[-1]];ev=p['events'];assert len(ds)==1 and np.all(ev[:,2]==1);amount=ds.NDOSE.iloc[0]*dem[dem.ID==ds.ID.iloc[0]].WEIGHT.iloc[0]*1000/130.08;assert abs(ev[0,1]-amount)<1e-9
 episodes.append(dict(profile_index=i,source_id=p['source_id'],drug_id=p['drug_id'],participant_id=p['participant_id'],profile_id=p['profile_id'],support_ids=tuple(h.observation_id),query_ids=tuple(q.observation_id),support_end=float(h.time.max()),query_first=float(q.time.min()),queries=len(q)))
pd.DataFrame(s5records).to_csv(S/'S5_records_LOCAL_ONLY.csv',index=False);pd.DataFrame(episodes).to_csv(S/'episode_audit_LOCAL_ONLY.csv',index=False)
s5=pd.DataFrame(s5records);save(S/'S5_conversion_verification.json',dict(raw_rows=len(c),canonical_rows=len(g),unretained_zero_rows=len(missing),positive_records=int((g.target_dv>0).sum()),checks=checks,molecular_weight=130.08,route='IV bolus; distributor records 0.5 minute administration duration; bolus approximation retained',roles=s5.groupby(['role','use']).size().to_dict().__str__(),clinical_scope='Distributor describes plasma; original assay and primary-record chain not established. Not a blanket clinical QC pass.',raw_reference_manifest=sha(raw/'manifest.json')))
# Primary pooled comparison and the current single-drug Table 1.
pooled=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet',filters=[('k','==',2)]);pooled=pooled[pooled.condition.isin(['molecular_relation','dcm','unipk','deepct','poppk_vi_map'])].copy();pooled['panel']='pooled'
single=pd.read_parquet(R/'results/summaries/training_scope_extension_20261001/single_drug_predictions.parquet',filters=[('k','==',2)]);single['panel']='single';single['track']='Known-drug'
def resolve(rid):
 if rid.startswith('results/'):return rid
 if rid.startswith('full__s'):return 'results/runs/TRACEABLE_POPPK/evaluation_20260925/'+rid
 return 'results/runs/TRACEABLE/evaluation_20260925/'+rid
pooled['run_id']=pooled.run_id.map(resolve);single['run_id']=single.run_id.map(resolve)
d=pd.concat([pooled,single],ignore_index=True);querychecks=[]
for (panel,track,seed),z in d.groupby(['panel','track','seed']):
 ref=None
 for cond,h in z.groupby('condition'):
  h=h.set_index('observation_id').sort_index();assert h.index.is_unique
  cols=['source_id','drug_id','participant_id','profile_id','query_time','y_true_log']
  if ref is None:ref=h
  else:assert h.index.equals(ref.index) and np.array_equal(h[cols].to_numpy(),ref[cols].to_numpy())
  assert np.isfinite(h[['y_true_log','y_pred_log']]).all().all() and np.isfinite(np.exp(h.y_pred_log)).all()
  querychecks.append(dict(panel=panel,track=track,seed=seed,condition=cond,queries=len(h),passed=True))
pd.DataFrame(querychecks).to_csv(S/'E02_query_checks.csv',index=False)
runrows=[];fits=[]
for rid,z in d.groupby('run_id'):
 p=R/rid;manifest=json.loads((p/'run_manifest.json').read_text());assert manifest['status']=='completed';hp=sha(p/'predictions.parquet');assert hp==manifest['prediction_sha256']
 rawp=pd.read_parquet(p/'predictions.parquet').rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});rawp=rawp[(rawp.k==2)&(rawp.protocol=='first_k')];rawp=rawp.set_index('observation_id');zz=z.drop_duplicates('observation_id').set_index('observation_id');assert zz.index.is_unique and rawp.index.is_unique;assert np.array_equal(rawp.loc[zz.index,['y_true_log','y_pred_log']],zz[['y_true_log','y_pred_log']])
 sel=json.loads((p/'selection.json').read_text());art=json.loads((p/'artifacts.json').read_text());pop='poppk' in rid.lower();v=pd.read_csv(p/'validation.csv');t=pd.read_csv(p/'training.csv');cfg=json.loads((p/'config.resolved.json').read_text())
 for drug,selection in (sel.items() if pop else [(None,sel)]):
  a=art[drug] if pop else art;partition=[set(map(tuple,meta.iloc[a[k]][['source_id','participant_id']].values.tolist())) for k in ['train','validation','test']];assert not(partition[0]&partition[1] or partition[0]&partition[2] or partition[1]&partition[2])
  if not pop:
   cal=json.loads((p/'population.json').read_text());assert cal['fitting_indices']==a['train'];assert set(a['scaler']['fitted_drugs'])==set(meta.iloc[a['train']].drug_id)
  ck=p/(drug+'_selected.pt' if pop else 'selected.pt');assert sha(ck)==selection['checkpoint_sha256']
  vv=v[v.drug==drug] if pop and 'drug' in v else v
  if pop:vv=vv[vv.epoch>0] # Original PopPK explicitly excludes epoch zero from selection.
  # Different population log schemas remain explicit.
  col='validation_sb_hrmse' if 'validation_sb_hrmse' in vv else None
  chosen=selection['epoch'];best=int(vv.loc[vv[col].idxmin(),'epoch']) if col else None
  if best is not None:
   chosen_value=float(vv[vv.epoch==chosen][col].iloc[0]);assert abs(chosen_value-float(vv[col].min()))<1e-12,(rid,drug,chosen,best,chosen_value,float(vv[col].min()))
  fits.append(dict(run_id=rid,drug_id=drug,condition=z.condition.iloc[0],fold=z.fold.iloc[0],seed=int(z.seed.iloc[0]),chosen_epoch=chosen,candidate_min=int(vv.epoch.min()),candidate_max=int(vv.epoch.max()),selected_at_max=chosen==vv.epoch.max(),checkpoint_sha256=sha(ck),validation_final=float(vv[col].iloc[-1]) if col else None,validation_best=float(vv[col].min()) if col else None,validation_improvement_last_window=float(vv[col].iloc[:-1].min()-vv[col].min()) if col and len(vv)>1 else None,weight_change_squared=selection.get('weight_change_squared'),training_rows=len(t),training_diagnostics='stored losses/gradients' if not pop else 'population VI training log'))
 runrows.append(dict(run_id=rid,status='completed',prediction_sha256=hp,config_sha256=sha(p/'config.resolved.json'),queries=len(zz),fallback_rows=int(rawp.fallback_flag.sum()) if 'fallback_flag' in rawp else None,numerical_floor_internal_count='not logged; final output at floor is not a count of internal floor applications'))
 if len(runrows)%100==0:print('audited',len(runrows),flush=True)
pd.DataFrame(runrows).to_csv(S/'E02_run_audit.csv',index=False);pd.DataFrame(fits).to_csv(S/'E02_training_audit.csv',index=False)
# Recalculate per-drug results and compare the tables backing current manuscript.
rows=[]
for key,z in d.groupby(['panel','track','condition','seed','drug_id']):rows.append(dict(zip(['panel','track','condition','seed','drug_id'],key))|balanced_metrics(z.y_true_log,z.y_pred_log,z.participant_id))
f=pd.DataFrame(rows);f.to_csv(S/'E02_drug_metrics_by_seed.csv',index=False)
expected_p=pd.read_csv(R/'results/summaries/ten_seeds_20261001/drug_metrics_summary.csv').query('k==2');expected_s=pd.read_csv(R/'results/summaries/training_scope_extension_20261001/single_drug_summary.csv').query('k==2');tablechecks=[]
for (panel,track,c,drug),h in f.groupby(['panel','track','condition','drug_id']):
 e=expected_p[(expected_p.track==track)&(expected_p.condition==c)&(expected_p.drug_id==drug)] if panel=='pooled' else expected_s[(expected_s.condition==c)&(expected_s.drug_id==drug)]
 value=float(e.iloc[0]['rmse' if panel=='pooled' else 'rmse_mean']);delta=float(h.log_rmse.mean()-value);assert abs(delta)<1e-10;tablechecks.append(dict(panel=panel,track=track,condition=c,drug_id=drug,recomputed=float(h.log_rmse.mean()),reported=value,difference=delta))
pd.DataFrame(tablechecks).to_csv(S/'E02_table_checks.csv',index=False)
# Check purported tied Mavoglurant predictions without rounding.
mt=[]
for (panel,track,seed),h in d[(d.drug_id=='mavoglurant')&d.condition.isin(['deepct','unipk'])].groupby(['panel','track','seed']):
 a=h[h.condition=='deepct'].set_index('observation_id').sort_index();b=h[h.condition=='unipk'].set_index('observation_id').sort_index();mt.append(dict(panel=panel,track=track,seed=seed,max_prediction_difference=float(abs(a.y_pred_log-b.y_pred_log).max()),identical_predictions=bool(np.array_equal(a.y_pred_log,b.y_pred_log)),deepct_run=a.run_id.iloc[0],unipk_run=b.run_id.iloc[0]))
pd.DataFrame(mt).to_csv(S/'E02_mavoglurant_ties.csv',index=False)
save(S/'E02_status.json',dict(status='completed',run_directories=len(runrows),model_fits=len(fits),table_cells=len(tablechecks),query_contract_checks=len(querychecks),max_table_difference=max(abs(x['difference']) for x in tablechecks),numerical_runtime_limit='Internal solver failure/floor telemetry not present in every historical run; final finite output and stored fallback flags verified. No complete convergence or native-reproduction claim.',script_sha256=sha(Path(__file__))))
print('E02 complete',len(runrows),len(fits))
