"""Prediction-derived additional comparisons, all drug-level metrics retained."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.drug_metrics import balanced_metrics
from mhap_pk.evaluation.quality_metrics import extended_metrics
B=R/'results/runs/COMPLETION_SUITE/20261002';O=R/'results/summaries/completion_suite_20261002';O.mkdir(exist_ok=True);torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(pd.read_csv(O/'centered_augmentation_verification.csv'))==100
assert json.loads((O/'primary_diagnostics_manifest.json').read_text())['query_checks']
def original_path(rid):return R/rid if rid.startswith('results/') else R/'results/runs/TRACEABLE/evaluation_20260925'/rid
base=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet');base=base[(base.protocol=='first_k')&base.condition.isin(['molecular_relation','dcm','unipk','deepct','poppk_vi_map'])].copy();base['model']=base.condition.map(lambda x:x+'_120' if x!='poppk_vi_map' else x);frames=[base];inventory=[];parities=[]
# Older PopPK exports omitted cutoff_time, but recorded support_end_time.
# Initial-observation cutoff is the last supplied observation; preserve raw metadata.
base['cutoff_time_recorded']=base.cutoff_time
missing=(base.k>0)&base.cutoff_time.isna()
assert base.loc[missing,'condition'].eq('poppk_vi_map').all()
assert base.loc[missing,'support_end_time'].notna().all()
base.loc[missing,'cutoff_time']=base.loc[missing,'support_end_time']
(O/'metadata_normalization.json').write_text(json.dumps(dict(condition='poppk_vi_map',protocol='first_k',field='cutoff_time',rows=int(missing.sum()),source_field='support_end_time',rule='Last supplied observation, matching initial-observation definition. No concentration, query, prediction, or training change.',raw_field_retained='cutoff_time_recorded'),indent=2))
run_lookup=base.groupby(['fold','seed','condition'],sort=False).run_id.first().to_dict()
artifact_cache={}
extra=json.loads((B/'budget_seed1_k3_verified/manifest.json').read_text());assert extra['status']=='completed';replay=pd.read_csv(B/'budget_seed1_k3_verified/replay_checks.csv');assert len(replay)==40;assert replay.max_prediction_difference.max()<1e-10;replay.to_csv(O/'seed1_k3_replay_checks.csv',index=False)
for name,expected in extra['files'].items():assert sha(B/'budget_seed1_k3_verified'/name)==expected
plan=json.loads((B/'registered_plan.json').read_text());jobs=plan['budget_jobs']+[dict(fold=f,seed=1,mode=m,reused=True) for f in sorted(base.fold.unique()) for m in ['molecular_relation','dcm','unipk','deepct']]
for j in jobs:
 root=(R/'results/runs/COMPARISON_REFRESH/20260929/budget240') if j.get('reused') else B/'budget240';p=root/f"{j['fold']}__s{j['seed']}__{j['mode']}";manifest=json.loads((p/'run_manifest.json').read_text());assert manifest['status']=='completed';assert sha(p/'predictions.parquet')==manifest['prediction_sha256'];s=json.loads((p/'selection.json').read_text());old=original_path(run_lookup[(j['fold'],j['seed'],j['mode'])]);a=torch.load(p/'selected_at120.pt',weights_only=True);b=torch.load(old/'selected.pt',weights_only=True);assert set(a)==set(b);assert all(torch.isfinite(v).all() for v in list(a.values())+list(b.values()));delta=max(float((a[k]-b[k]).abs().max()) for k in a);assert delta<1e-10;parities.append(j|dict(max_parameter_difference=delta,parent=str(old.relative_to(R))))
 d=pd.read_parquet(p/'predictions.parquet').rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});d=d[d.protocol=='first_k'].copy();d=pd.concat([d,pd.read_parquet(B/'budget_seed1_k3_verified'/f"{j['fold']}__{j['mode']}.parquet")],ignore_index=True) if j.get('reused') and 3 not in set(d.k) else d;d['run_id']=str(p.relative_to(R));d['model']=j['mode']+'_240';d['track']=np.where(d.fold=='known','Known-drug','Joint drug–source');frames.append(d);inventory.append(dict(family='budget',model=d.model.iloc[0],fold=j['fold'],seed=j['seed'],epoch=s['epoch'],parameters=s['parameters'],elapsed_seconds=manifest['elapsed_seconds'],run=str(p.relative_to(R)),prediction_sha256=sha(p/'predictions.parquet')))
for j in json.loads((B/'controls_plan.json').read_text())['jobs']+json.loads((B/'residual_augmentation_plan.json').read_text())['jobs']:
 p=B/j['variant']/f"{j['fold']}__s{j['seed']}__molecular_relation";manifest=json.loads((p/'run_manifest.json').read_text());assert manifest['status']=='completed';assert sha(p/'predictions.parquet')==manifest['prediction_sha256'];s=json.loads((p/'selection.json').read_text());d=pd.read_parquet(p/'predictions.parquet').rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});d=d[d.protocol=='first_k'].copy();d['run_id']=str(p.relative_to(R));d['model']=j['variant'];d['track']=np.where(d.fold=='known','Known-drug','Joint drug–source');frames.append(d);inventory.append(dict(family='controls',model=j['variant'],fold=j['fold'],seed=j['seed'],epoch=s['epoch'],parameters=s['parameters'],elapsed_seconds=manifest['elapsed_seconds'],run=str(p.relative_to(R)),prediction_sha256=sha(p/'predictions.parquet')))
audit=[];cost=[]
for rec in inventory:
 p=R/rec['run'];art=json.loads((p/'artifacts.json').read_text());ref_run=run_lookup[(rec['fold'],rec['seed'],'molecular_relation')];ref_art=artifact_cache.setdefault(ref_run,json.loads((original_path(ref_run)/'artifacts.json').read_text()));assert art==ref_art;pop=json.loads((p/'population.json').read_text());assert pop['fitting_indices']==art['train'];v=pd.read_csv(p/'validation.csv');sel=json.loads((p/'selection.json').read_text());assert sha(p/'selected.pt')==sel['checkpoint_sha256'];a=dict(rec,checkpoint_sha256=sel['checkpoint_sha256'],resolved_config_sha256=sha(p/'config.resolved.json'),fitting_artifact_sha256=sha(p/'artifacts.json'),population_sha256=sha(p/'population.json'),validation_best=sel['validation_sb_hrmse'],fitting_partition_and_scaler_match=True,population_fitted_on_training=True,initial_selected=sel['epoch']==0,ceiling_selected=sel['epoch']==int(v.epoch.max()))
 if rec['family']=='budget':
  old=json.loads((p/'selection_at120.json').read_text());a.update(epoch_at120=old['epoch'],validation_at120=old['validation_sb_hrmse'],validation_improvement=old['validation_sb_hrmse']-sel['validation_sb_hrmse'])
 audit.append(a)
 for row in pd.read_csv(p/'latency.csv').to_dict('records'):
  cost.append(dict(model=rec['model'],fold=rec['fold'],seed=rec['seed'],**row,measurement='two forward calls, original plus checkpoint replay; batch timing'))
pd.DataFrame(audit).to_csv(O/'optimization_audit.csv',index=False);pd.DataFrame(cost).to_csv(O/'inference_timing_audit.csv',index=False)
pd.DataFrame(audit).groupby(['model','fold']).agg(fits=('seed','size'),selected_epoch_median=('epoch','median'),selected_at_initialization=('initial_selected','sum'),selected_at_ceiling=('ceiling_selected','sum'),median_wall_seconds=('elapsed_seconds','median')).reset_index().to_csv(O/'optimization_summary.csv',index=False)
allp=pd.concat(frames,ignore_index=True);assert allp.run_id.notna().all();assert not allp.duplicated(['model','seed','fold','k','observation_id']).any();assert np.isfinite(allp[['y_true_log','y_pred_log']]).all().all()
assert (allp.loc[allp.k>0,'query_time']>allp.loc[allp.k>0,'support_end_time']).all()
checks=[];queries_across_k={}
for (track,seed,k),g in allp.groupby(['track','seed','k']):
 reference=None
 for model,h in g.groupby('model'):
  x=h.set_index(['source_id','drug_id','participant_id','profile_id','observation_id'])[['query_time','y_true_log','support_end_time','cutoff_time']].sort_index()
  if reference is None:reference=x
  else:pd.testing.assert_frame_equal(reference,x)
 core=reference[['query_time','y_true_log']]
 if (track,seed) in queries_across_k:pd.testing.assert_frame_equal(queries_across_k[(track,seed)],core)
 else:queries_across_k[(track,seed)]=core
 checks.append(dict(track=track,seed=seed,k=k,models=g.model.nunique(),queries=len(reference)))
assert allp.groupby(['track','model','k']).seed.nunique().eq(10).all()
allp.to_parquet(O/'canonical_predictions_LOCAL_ONLY.parquet',index=False);pd.DataFrame(checks).to_csv(O/'query_checks.csv',index=False);pd.DataFrame(inventory).to_csv(O/'new_run_inventory.csv',index=False);pd.DataFrame(parities).to_csv(O/'budget120_replay_checks.csv',index=False)
rows=[];cohort=[]
for key,g in allp.groupby(['track','model','seed','k','source_id','drug_id']):rows.append(dict(zip(['track','model','seed','k','source_id','drug_id'],key))|balanced_metrics(g.y_true_log,g.y_pred_log,g.source_id+'::'+g.participant_id))
for key,g in allp.groupby(['track','model','seed','k']):
 x=g.copy();x['condition']=key[1];cohort.append(dict(zip(['track','model','seed','k'],key))|extended_metrics(x))
d=pd.DataFrame(rows);c=pd.DataFrame(cohort);d.to_csv(O/'drug_metrics_by_seed.csv',index=False);c.to_csv(O/'cohort_metrics_by_seed.csv',index=False)
fields=['log_rmse','log_mae','within_twofold','signed_log_bias','linear_rmse','linear_mae','r2_log','r2_linear'];ds=d.groupby(['track','model','k','drug_id'])[fields].agg(['mean','std']);ds.columns=['_'.join(x) for x in ds.columns];ds=ds.reset_index();counts=d.groupby(['track','model','k','drug_id'])[['participants','queries']];assert counts.nunique().eq(1).all().all();ds=ds.merge(counts.first().reset_index(),on=['track','model','k','drug_id'],validate='one_to_one');ds.to_csv(O/'drug_summary.csv',index=False)
cs=c.groupby(['track','model','k'])[['sb_hrmse','log_rmse','log_mae','within_twofold_pct','signed_bias']].agg(['mean','std']);cs.columns=['_'.join(x) for x in cs.columns];cs=cs.reset_index();cs.to_csv(O/'cohort_summary.csv',index=False)
# Synthetic experiment held-out evaluation, not human-cohort measurements.
sims=[]
for seed in range(1,11):
 p=B/'simulation_crossed'/f'seed{seed}';assert json.loads((p/'manifest.json').read_text())['status']=='completed';sims.append(pd.read_csv(p/'metrics.csv'))
sim=pd.concat(sims,ignore_index=True);sim.to_csv(O/'simulation_by_seed.csv',index=False);su=sim.groupby(['model','prior_shift','noise','timing','omitted_dose','k'])[['latent_log_rmse','observed_log_rmse','harm_fraction','mse_difference']].agg(['mean','std']);su.columns=['_'.join(x) for x in su.columns];su=su.reset_index();su.to_csv(O/'simulation_summary.csv',index=False)
with pd.ExcelWriter(O/'Completion_results.xlsx') as w:
 cs.to_excel(w,sheet_name='Cohort',index=False);ds.to_excel(w,sheet_name='Drug',index=False);d.to_excel(w,sheet_name='Drug_seed',index=False);su.to_excel(w,sheet_name='Synthetic_mechanism',index=False);pd.DataFrame(inventory).to_excel(w,sheet_name='Fits',index=False)
 for f,n in [('resampling_summary.csv','Resampling'),('stage_summary.csv','Update_stages'),('augmentation_teacher_fit.csv','Augmentation_teacher')]:pd.read_csv(O/f).to_excel(w,sheet_name=n,index=False)
import subprocess
subprocess.run([sys.executable,str(R/'scripts/plot_completion_summary.py')],check=True)
initial=pd.concat([pd.read_csv(B/'simulation'/f'seed{seed}'/'metrics.csv') for seed in range(1,11)],ignore_index=True)
initial.to_csv(O/'simulation_initial_confounded_training_by_seed.csv',index=False)
(O/'completion_manifest.json').write_text(json.dumps(dict(status='completed',canonical_prediction_sha256=sha(O/'canonical_predictions_LOCAL_ONLY.parquet'),input_prediction_sha256=sha(R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet'),registered_plan_sha256=sha(B/'registered_plan.json'),controls_plan_sha256=sha(B/'controls_plan.json'),centered_augmentation_plan_sha256=sha(B/'residual_augmentation_plan.json'),new_fits=360+500+30+30+100,reused_budget_fits=40,simulation_replicates=10,additional_initial_simulation_fits=30,all_query_checks_passed=True,original_120epoch_checkpoint_parity=True,new_primary_model_selected=False,files={p.name:sha(p) for p in O.iterdir() if p.is_file() and 'LOCAL_ONLY' not in p.name and p.name!='completion_manifest.json'}),indent=2));print(cs[cs.k==2].to_string(index=False))
