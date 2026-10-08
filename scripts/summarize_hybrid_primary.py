"""Aggregate only adopted-model reruns; retain explicit scope and provenance."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from mhap_pk.evaluation.quality_metrics import extended_metrics
from mhap_pk.evaluation.drug_metrics import balanced_metrics
from summarize_training_scope_extension import summarize
from mhap_pk.data.traceable import load_profiles
B=R/'results/runs/HYBRID_PRIMARY/20261002';S=R/'results/summaries/hybrid_primary_20261002';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 plan=json.loads((B/'registered_plan.json').read_text());status=json.loads((B/'status.json').read_text());assert status['status']=='completed',status['status']
 for p,h in json.loads((B/'code_manifest.json').read_text()).items():assert sha(R/p)==h,p
 ps,meta,_=load_profiles();frames=[];inv=[];checks=[]
 for j in plan['jobs']:
  mode={'chemical_only':'chemical_only','pk_only':'pk_only','both_relations':'molecular_relation'}.get(j['variant'],'molecular');root=R/j['root']/f"{j['fold']}__s{j['seed']}__{mode}";m=json.loads((root/'run_manifest.json').read_text());assert m['status']=='completed';assert sha(root/'predictions.parquet')==m['prediction_sha256'];sel=json.loads((root/'selection.json').read_text());assert sha(root/'selected.pt')==sel['checkpoint_sha256']
  a=json.loads((root/'artifacts.json').read_text());cal=json.loads((root/'population.json').read_text());assert a['train']==cal['fitting_indices']
  roles=[set(map(tuple,meta.iloc[a[k]][['source_id','participant_id']].to_numpy())) for k in ['train','validation','test']];assert not(roles[0]&roles[1] or roles[0]&roles[2] or roles[1]&roles[2])
  assert set(a['scaler']['fitted_drugs'])==set(meta.iloc[a['train']].drug_id)
  if j['fold']!='known':assert set(meta.iloc[a['train']].drug_id).isdisjoint(set(meta.iloc[a['test']].drug_id))
  parentcal=json.loads((R/j['parent']/'calibration'/j['fold']/'population.json').read_text());assert cal['log_parameters']==parentcal['log_parameters']
  ver=json.loads((root/'verification.json').read_text());assert ver['max_prediction_difference']<1e-10
  p=pd.read_parquet(root/'predictions.parquet').rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});p=p[p.protocol=='first_k'].copy();p['condition']=j['variant'];p['kind']=j['kind'];p['track']='Known-drug' if j['fold']=='known' else 'Joint drug–source';p['run_id']=str(root.relative_to(R));p['scope']=j.get('scope',j['kind']);p['target_drug']=j.get('drug','');p['cap']=j.get('cap',0);p['evaluation_partition']=j.get('partition','');frames.append(p)
  checks.append(dict(run_id=str(root.relative_to(R)),partition_disjoint=True,fitted_drugs_checked=True,calibration_checked=True,max_replay_difference=ver['max_prediction_difference']))
  inv.append(j|dict(run_id=str(root.relative_to(R)),prediction_sha256=sha(root/'predictions.parquet'),config_sha256=sha(root/'config.resolved.json'),seconds=m['elapsed_seconds'],**sel))
 p=pd.concat(frames,ignore_index=True);assert np.isfinite(p[['y_true_log','y_pred_log']]).all().all() and np.isfinite(np.exp(p.y_pred_log)).all();p.to_parquet(S/'new_analysis_predictions.parquet',index=False)
 inv=pd.DataFrame(inv);inv.to_csv(S/'new_run_inventory.csv',index=False);pd.DataFrame(checks).to_csv(S/'new_run_checks.csv',index=False)
 primary=pd.read_parquet(S/'primary_predictions.parquet');ours=primary[(primary.condition=='ours')&(primary.protocol=='first_k')].copy();ours['kind']='panel'
 # Panel ablations and primary model always use the identical queried profiles.
 panel=pd.concat([p[p.kind=='panel'],ours],ignore_index=True)
 qc=[];keys=['source_id','drug_id','participant_id','profile_id','observation_id']
 for (track,seed,k),g in panel.groupby(['track','seed','k']):
  ref=g[g.condition=='ours'].set_index(keys)[['query_time','y_true_log','support_end_time']].sort_index()
  for model,h in g.groupby('condition'):
   pd.testing.assert_frame_equal(ref,h.set_index(keys)[ref.columns].sort_index());qc.append(dict(track=track,seed=seed,k=int(k),model=model))
 def metricdata(frame,cols):return pd.DataFrame([dict(zip(cols,key),**balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id+'::'+g.participant_id)) for key,g in frame.groupby(cols,dropna=False)])
 def ds(frame,cols):
  fields=['log_rmse','log_mae','within_twofold','signed_log_bias','linear_rmse','linear_mae','r2_log','r2_linear'];return summarize(frame,cols,fields)
 d=metricdata(panel,['track','condition','seed','k','source_id','drug_id']);d.to_csv(S/'ablation_drug_by_seed.csv',index=False);ds(d,['track','condition','k','source_id','drug_id','participants','queries']).to_csv(S/'ablation_drug_summary.csv',index=False)
 c=pd.DataFrame([dict(track=t,condition=m,seed=s,k=k,**extended_metrics(g)) for (t,m,s,k),g in panel.groupby(['track','condition','seed','k'])]);c.to_csv(S/'ablation_cohort_by_seed.csv',index=False);summarize(c,['track','condition','k'],[v for v in c if v not in ['track','condition','seed','k']]).to_csv(S/'ablation_cohort_summary.csv',index=False)
 pd.DataFrame(qc).to_csv(S/'ablation_query_checks.csv',index=False)
 # Full target-only fits use target-specific validation and supply the full-budget target arm.
 single=p[p.kind=='single'].copy();single['condition']='ours'
 oldsingle=pd.read_parquet(R/'results/summaries/training_scope_extension_20261001/single_drug_predictions.parquet');oldsingle=oldsingle[oldsingle.condition.isin(['dcm','unipk','deepct','poppk_vi_map'])].copy();oldsingle['track']='Known-drug';single=pd.concat([single,oldsingle],ignore_index=True)
 for (seed,k,drug),g in single.groupby(['seed','k','drug_id']):
  ref=g[g.condition=='ours'].set_index(keys)[['query_time','y_true_log']].sort_index()
  for model,h in g.groupby('condition'):pd.testing.assert_frame_equal(ref,h.set_index(keys)[ref.columns].sort_index())
 single.to_parquet(S/'single_predictions.parquet',index=False);d=metricdata(single,['condition','seed','k','source_id','drug_id']);d.to_csv(S/'single_drug_by_seed.csv',index=False);ds(d,['condition','k','source_id','drug_id','participants','queries']).to_csv(S/'single_drug_summary.csv',index=False)
 low=p[p.kind=='low_data'].copy();full=p[p.kind=='single'].copy();full['scope']='target'
 counts=pd.read_csv(R/'results/summaries/single_drug_20261001/partition_counts.csv').set_index('drug_id');full['cap']=full.drug_id.map(counts.train_participants);low=pd.concat([low,full],ignore_index=True)
 for (seed,k,drug,cap),g in low.groupby(['seed','k','target_drug','cap']):
  ref=g[g.scope=='target'].set_index(keys)[['query_time','y_true_log']].sort_index();other=g[g.scope=='pooled'].set_index(keys)[ref.columns].sort_index();pd.testing.assert_frame_equal(ref,other)
 low.to_parquet(S/'pooling_predictions.parquet',index=False);d=metricdata(low,['scope','cap','seed','k','source_id','drug_id']);d.to_csv(S/'pooling_drug_by_seed.csv',index=False);ds(d,['scope','cap','k','source_id','drug_id','participants','queries']).to_csv(S/'pooling_drug_summary.csv',index=False)
 key=['cap','seed','k','source_id','drug_id'];a=d[d.scope=='pooled'].merge(d[d.scope=='target'],on=key,suffixes=('','_target'),validate='one_to_one');a['delta_log_rmse']=a.log_rmse-a.log_rmse_target;a.to_csv(S/'pooling_paired_by_seed.csv',index=False);summarize(a,['cap','k','drug_id'],['delta_log_rmse']).to_csv(S/'pooling_paired_summary.csv',index=False)
 # CV concatenates OOF queries before computing metrics; never average fold RMSEs.
 cv=p[p.kind=='resampling'].copy();cv['evaluation']=np.where(cv.evaluation_partition.str.startswith('cv5'),'CV5',cv.evaluation_partition);cv['condition']='ours'
 oldcv=pd.read_parquet(R/'results/summaries/resampling_20261002/canonical_predictions.parquet',filters=[('training_scope','==','pooled'),('protocol','==','first_k')]);oldcv=oldcv[oldcv.model.isin(['dcm','deepct','unipk','poppk_vi_map'])].copy();oldcv['condition']=oldcv.model
 cv=pd.concat([cv,oldcv],ignore_index=True)
 for (evaluation,seed,k),g in cv.groupby(['evaluation','seed','k']):
  ref=g[g.condition=='ours'].set_index(keys)[['query_time','y_true_log']].sort_index();assert ref.index.is_unique
  for model,h in g.groupby('condition'):pd.testing.assert_frame_equal(ref,h.set_index(keys)[ref.columns].sort_index())
  if evaluation=='CV5':assert set(ref.index.get_level_values('observation_id').astype(str))=={str(q) for pr in ps for q in pr['query'].observation_id}
 cv.to_parquet(S/'resampling_predictions.parquet',index=False);d=metricdata(cv,['evaluation','condition','seed','k','source_id','drug_id']);d.to_csv(S/'resampling_drug_by_seed.csv',index=False);ds(d,['evaluation','condition','k','source_id','drug_id','participants','queries']).to_csv(S/'resampling_drug_summary.csv',index=False)
 c=pd.DataFrame([dict(evaluation=e,condition=m,seed=s,k=k,**extended_metrics(g)) for (e,m,s,k),g in cv.groupby(['evaluation','condition','seed','k'])]);c.to_csv(S/'resampling_cohort_by_seed.csv',index=False);summarize(c,['evaluation','condition','k'],[v for v in c if v not in ['evaluation','condition','seed','k']]).to_csv(S/'resampling_cohort_summary.csv',index=False)
 # Equal-budget comparators are unchanged; match seeds/folds/query sets with the new 240-epoch model.
 budget=p[(p.kind=='panel')&(p.condition=='budget240')].copy();budget['condition']='ours_240';oldb=pd.read_parquet(R/'results/summaries/completion_suite_20261002/canonical_predictions_LOCAL_ONLY.parquet') if (R/'results/summaries/completion_suite_20261002/canonical_predictions_LOCAL_ONLY.parquet').exists() else None
 if oldb is not None:
  # Existing completion table may use model rather than condition.
  if 'model' in oldb:oldb['condition']=oldb.model
  oldb=oldb[oldb.condition.isin(['dcm_240','unipk_240','deepct_240'])&(oldb.protocol=='first_k')];budget=pd.concat([budget,oldb],ignore_index=True)
 for (track,seed,k),g in budget.groupby(['track','seed','k']):
  ref=g[g.condition=='ours_240'].set_index(keys)[['query_time','y_true_log']].sort_index()
  for model,h in g.groupby('condition'):pd.testing.assert_frame_equal(ref,h.set_index(keys)[ref.columns].sort_index())
 budget.to_parquet(S/'budget_predictions.parquet',index=False)
 c=pd.DataFrame([dict(track=t,condition=m,seed=s,k=k,**extended_metrics(g)) for (t,m,s,k),g in budget.groupby(['track','condition','seed','k'])]);c.to_csv(S/'budget_cohort_by_seed.csv',index=False);summarize(c,['track','condition','k'],[v for v in c if v not in ['track','condition','seed','k']]).to_csv(S/'budget_cohort_summary.csv',index=False)
 d=metricdata(budget,['track','condition','seed','k','source_id','drug_id']);d.to_csv(S/'budget_drug_by_seed.csv',index=False)
 (S/'analysis_manifest.json').write_text(json.dumps(dict(status='completed',new_fits=len(inv),primary_reused_fits=100,primary_model='hybrid/no relations',ablation_query_checks=len(qc),all_new_checkpoint_parities_verified=True,source_manifest=sha(B/'code_manifest.json'),primary_predictions=sha(S/'primary_predictions.parquet'),new_predictions=sha(S/'new_analysis_predictions.parquet')),indent=2))
 with pd.ExcelWriter(S/'MoDE_PK_hybrid_results.xlsx') as w:
  for name in ['primary_drug_summary','primary_cohort_summary','ablation_drug_summary','ablation_cohort_summary','single_drug_summary','pooling_drug_summary','pooling_paired_summary','resampling_drug_summary','resampling_cohort_summary','primary_source_bootstrap']:
   pd.read_csv(S/(name+'.csv')).to_excel(w,sheet_name=name[:31],index=False)
 print('Summarized',len(inv),'new fits; all current Ours analyses use hybrid checkpoints.')
if __name__=='__main__':main()
