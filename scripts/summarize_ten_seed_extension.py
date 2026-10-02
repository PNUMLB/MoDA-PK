"""Verified 10-seed main, drug, history and ablation tables. No score selection."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.quality_metrics import extended_metrics
B=R/'results/runs/TEN_SEEDS/20261001';O=R/'results/summaries/ten_seeds_20261001';O.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
status=json.loads((B/'status.json').read_text());assert status['status']=='completed'
plan=json.loads((B/'registered_plan.json').read_text());assert len(status['jobs'])==810
old_history=R/'results/summaries/history_count_20261001/canonical_predictions.parquet'
old_ablation=R/'results/summaries/comparison_refresh_20260929/ablation_predictions.parquet'
history=[pd.read_parquet(old_history)];ablation=[pd.read_parquet(old_ablation)];inventory=[]
main_modes=['molecular_relation','molecular','dcm','deepct','unipk','poppk_vi_map','single_drug']
for j in status['jobs']:
 assert j['status']=='completed'
 family=j['family']
 if family=='single':run=B/'single'/j['drug']/f"known__s{j['seed']}__molecular"
 elif family=='poppk':run=B/'poppk'/f"full__s{j['seed']}"
 else:run=B/family/f"{j['fold']}__s{j['seed']}__{j['mode']}"
 f=run/'predictions.parquet';m=json.loads((run/'run_manifest.json').read_text());assert m['status']=='completed' and sha(f)==m['prediction_sha256']
 p=pd.read_parquet(f,filters=[('protocol','==','first_k')]).rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'})
 assert set(p.seed)=={j['seed']} and set(p.k)=={0,1,2,3,4}
 p['track']=np.where(p.fold=='known','Known-drug','Joint drug–source');p['run_id']=str(run.relative_to(R));p['dataset_id']='source_traceable_20260925';p['split_id']='source_traceable_20260925'
 if family=='single':p['condition']='single_drug'
 elif family=='components':p['condition']='component_'+p.condition
 if p.condition.iloc[0] in main_modes:history.append(p)
 if family=='components' or (family=='panel' and j['mode'] not in ['dcm','deepct','unipk']):ablation.append(p[p.k==2])
 checks=json.loads((run/'verification.json').read_text()) if (run/'verification.json').exists() else {}
 inventory.append(dict(run_id=str(run.relative_to(R)),family=family,condition=p.condition.iloc[0],fold=j['fold'],seed=j['seed'],drug=j['drug'],prediction_sha256=sha(f),fit_seconds=m.get('elapsed_seconds'),evaluated_first_k=sorted(p.k.unique().tolist()),actual_primary_query_rows=int((p.k==2).sum())))
p=pd.concat(history,ignore_index=True);ab=pd.concat(ablation,ignore_index=True);G=['track','condition','seed','k'];keys=['source_id','drug_id','participant_id','profile_id'];checks=[]
for (track,seed),g in p.groupby(['track','seed']):
 ref=g[(g.condition=='molecular_relation')&(g.k==2)].set_index('observation_id').sort_index()
 for (cond,k),h in g.groupby(['condition','k']):
  h=h.set_index('observation_id').sort_index();assert h.index.is_unique and h.index.equals(ref.index)
  assert np.array_equal(h.y_true_log,ref.y_true_log) and np.array_equal(h.query_time,ref.query_time)
  assert np.array_equal(h[keys].to_numpy(),ref[keys].to_numpy());checks.append(dict(track=track,seed=int(seed),condition=cond,k=int(k),identical_queries_targets=True,rows=len(h)))
assert set(p.seed)==set(range(1,11));assert np.isfinite(p[['y_true_log','y_pred_log']].to_numpy()).all()
for _,g in p.groupby(['track','condition','k']):assert set(g.seed)==set(range(1,11))
for (track,seed),g in ab.groupby(['track','seed']):
 ref=p[(p.track==track)&(p.seed==seed)&(p.condition=='molecular_relation')&(p.k==2)].set_index('observation_id').sort_index()
 for _,h in g.groupby('condition'):
  h=h.set_index('observation_id').sort_index();assert h.index.is_unique and h.index.equals(ref.index);assert np.array_equal(h.y_true_log,ref.y_true_log)
for _,g in ab.groupby(['track','condition']):assert set(g.seed)==set(range(1,11))
p.to_parquet(O/'canonical_predictions.parquet',index=False);ab.to_parquet(O/'ablation_predictions.parquet',index=False);pd.DataFrame(inventory).to_csv(O/'new_run_inventory.csv',index=False);pd.DataFrame(checks).to_csv(O/'query_checks.csv',index=False)
metrics=[]
for key,g in p.groupby(G):metrics.append(dict(zip(G,key))|extended_metrics(g))
m=pd.DataFrame(metrics);m.to_csv(O/'metrics_by_seed.csv',index=False)
metric_names=['sb_hrmse','log_rmse','log_mae','within_twofold_pct','signed_bias','p90_abs_log_error']
def summarize(frame,by):
 z=frame.groupby(by).agg({c:['mean','std','median'] for c in metric_names});z.columns=['_'.join(c) for c in z.columns];return z.reset_index()
s=summarize(m,['track','condition','k']);s.to_csv(O/'cohort_summary.csv',index=False)
ams=[]
for (track,c,seed),g in ab.groupby(['track','condition','seed']):ams.append(dict(track=track,condition=c,seed=seed,**extended_metrics(g)))
am=pd.DataFrame(ams);am.to_csv(O/'ablation_metrics_by_seed.csv',index=False);asm=summarize(am,['track','condition']);asm.to_csv(O/'ablation_summary.csv',index=False)
def drug_metrics(frame):
 frame=frame.copy();frame['sq']=(frame.y_pred_log-frame.y_true_log)**2;frame['ae']=abs(frame.y_pred_log-frame.y_true_log);frame['twofold']=100*(frame.ae<=np.log(2));frame['bias']=frame.y_pred_log-frame.y_true_log
 group=['track','condition','seed','k','source_id','drug_id'];part=frame.groupby(group+['participant_id']).agg(mse=('sq','mean'),mae=('ae','mean'),twofold=('twofold','mean'),bias=('bias','mean'),queries=('observation_id','size')).reset_index();d=part.groupby(group).agg(mse=('mse','mean'),mae=('mae','mean'),twofold=('twofold','mean'),bias=('bias','mean'),participants=('participant_id','size'),queries=('queries','sum')).reset_index();d['rmse']=np.sqrt(d.mse)
 a=d.groupby(['track','condition','drug_id','k']).agg(rmse=('rmse','mean'),rmse_sd=('rmse','std'),mae=('mae','mean'),mae_sd=('mae','std'),twofold=('twofold','mean'),twofold_sd=('twofold','std'),bias=('bias','mean'),participants=('participants','first'),queries=('queries','first')).reset_index();return d,a
perseed,drugs=drug_metrics(p);perseed.to_csv(O/'drug_metrics_by_seed.csv',index=False);drugs.to_csv(O/'drug_metrics_summary.csv',index=False)
ad,ads=drug_metrics(ab);ad.to_csv(O/'ablation_drug_metrics_by_seed.csv',index=False);ads.to_csv(O/'ablation_drug_metrics_summary.csv',index=False)
# Keep every seed and compare predefined blocks; never search best subsets.
blocks=[]
for name,seeds in [('seeds_1_5',range(1,6)),('seeds_6_10',range(6,11)),('seeds_1_10',range(1,11))]:
 z=summarize(m[m.seed.isin(seeds)],['track','condition','k']);z['seed_block']=name;blocks.append(z)
stability=pd.concat(blocks,ignore_index=True);stability.to_csv(O/'seed_block_stability.csv',index=False)
# All-source omission sensitivity at k2, preserving the complete primary table.
loo=[]
for (track,c,seed),g in perseed[perseed.k==2].groupby(['track','condition','seed']):
 for source in sorted(g.source_id.unique()):loo.append(dict(track=track,condition=c,seed=seed,omitted_source=source,sb_hrmse=g[g.source_id!=source].groupby('source_id').rmse.mean().mean()))
pd.DataFrame(loo).to_csv(O/'source_omission_by_seed.csv',index=False)
N={'molecular_relation':'MoDE-PK (Ours)','molecular':'Pooled MHAP-PK, relations OFF','single_drug':'Single-drug MHAP-PK','dcm':'DCM','unipk':'Uni-PK','deepct':'DeepCt','poppk_vi_map':'PopPK (MAP)'};core=['single_drug','molecular','molecular_relation'];panel=['molecular_relation','poppk_vi_map','dcm','unipk','deepct'];pretty=lambda d:'R-lipoic acid' if d=='r_lipoic_acid' else d.replace('_',' ').capitalize()
lines=['# Ten-seed experiment report','', 'Seeds 1–10 are all included. Seeds 1–5 reuse immutable results; seeds 6–10 add 860 model fits in 810 worker jobs. The separate seed-1 training replay is a parity check, not an additional replicate. No test-based seed or checkpoint selection.','']
for track in ['Known-drug','Joint drug–source']:
 mods=[c for c in panel if c!='poppk_vi_map' or track=='Known-drug'];lines+=['## '+track+' — primary k=2','', '| Drug | Participants | Future measurements | '+' | '.join(N[c] for c in mods)+' |','|---|---:|---:|'+'---:|'*len(mods)]
 z=drugs[(drugs.track==track)&(drugs.k==2)]
 for drug,g in z.groupby('drug_id'):
  g=g.set_index('condition');best=g.loc[mods].rmse.min();vals=[]
  for c in mods:
   v=f'{g.loc[c,"rmse"]:.4f} ± {g.loc[c,"rmse_sd"]:.4f}';vals.append('**'+v+'**' if g.loc[c,'rmse']==best else v)
  lines.append('| '+pretty(drug)+f' | {int(g.loc[mods[0],"participants"])} | {int(g.loc[mods[0],"queries"])} | '+' | '.join(vals)+' |')
 lines+=['','Values are participant-balanced drug log-RMSE, mean ± sample SD across ten seeds. Bold marks the lowest mean, not statistical significance.','']
for track,mods,title in [('Known-drug',core,'Single-drug versus pooled'),('Known-drug',panel,'Known-drug history count'),('Joint drug–source',[c for c in panel if c!='poppk_vi_map'],'Joint history count')]:
 lines+=['## '+title,'','| k | '+' | '.join(N[c] for c in mods)+' |','|---|'+'---:|'*len(mods)]
 for k in range(5):
  g=s[(s.track==track)&(s.k==k)].set_index('condition');lines.append('| '+str(k)+' | '+' | '.join(f'{g.loc[c,"sb_hrmse_mean"]:.4f} ± {g.loc[c,"sb_hrmse_std"]:.4f}' for c in mods)+' |')
 lines+=['']
lines+=['## Ablations at k=2','', '| Condition | Known-drug SB-hRMSE | Joint SB-hRMSE |','|---|---:|---:|']
for cond,g in asm.groupby('condition'):
 g=g.set_index('track');lines.append('| '+cond+' | '+' | '.join(f'{g.loc[t,"sb_hrmse_mean"]:.4f} ± {g.loc[t,"sb_hrmse_std"]:.4f}' for t in ['Known-drug','Joint drug–source'])+' |')
lines+=['','Scope: source-traceable cohort and current primary/ablation configurations. Historical fixed-prior, synthetic, and independent-data studies are not asserted to have ten-seed replication. First-k history retains the same future queries after t4; training samples k=0,1,2,4 and k=3 is additional inference. Common-origin k=0,1,2,4 predictions remain in the run directories. Molecular input/relation controls use the original fitting rules. PopPK and independent single-drug models are Known-only. All source omission and single-drug comparisons remain visible, including unfavorable results. Ten seeds quantify fitting variability, not ten independent cohorts or an external validation.']
(O/'REPORT.md').write_text('\n'.join(lines))
with pd.ExcelWriter(O/'ten_seed_results.xlsx',engine='openpyxl') as x:
 for frame,sheet in [(s,'Cohort metrics'),(drugs,'Drug metrics'),(m,'Cohort per seed'),(perseed,'Drug per seed'),(asm,'Ablation summary'),(ads,'Ablation drug metrics'),(am,'Ablation per seed'),(stability,'Seed block stability'),(pd.DataFrame(inventory),'New run inventory')]:frame.to_excel(x,sheet_name=sheet,index=False)
 for ws in x.book.worksheets:
  ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
  for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=min(36,max(12,len(str(col[0].value))+2))
C={'molecular_relation':'#b2182b','molecular':'#c17d11','single_drug':'#777777','dcm':'#8856a7','deepct':'#d95f02','unipk':'#1b7837','poppk_vi_map':'#2166ac'}
fig,axs=plt.subplots(1,3,figsize=(16,4.8))
for ax,track,mods,title in zip(axs,['Known-drug','Known-drug','Joint drug–source'],[core,panel,panel],['Known: single vs pooled','Known-drug comparators','Joint drug–source holdout']):
 for c in mods:
  g=s[(s.track==track)&(s.condition==c)].sort_values('k')
  if len(g):ax.errorbar(g.k,g.sb_hrmse_mean,yerr=g.sb_hrmse_std,marker='o',capsize=3,color=C[c],label=N[c])
 ax.set_title(title);ax.set_xticks(range(5));ax.set_xlabel('Initial observation count k');ax.set_ylabel('SB-hRMSE');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.suptitle('Ten seeds; identical future queries; mean ± seed SD');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(O/'history_comparison.pdf');fig.savefig(O/'history_comparison.png',dpi=150);plt.close(fig)
# Explicit cohort-level and per-drug pooling directions for interpretation.
effects=[]
for k,g in drugs[drugs.track=='Known-drug'].groupby('k'):
 w=g.pivot(index='drug_id',columns='condition',values='rmse');effects.append(dict(k=int(k),single_lower_than_full=int((w.single_drug<w.molecular_relation).sum()),full_lower_than_single=int((w.molecular_relation<w.single_drug).sum()),drugs=len(w)))
pd.DataFrame(effects).to_csv(O/'pooling_drug_win_counts.csv',index=False)
manifest=dict(status='completed',seeds=list(range(1,11)),new_worker_jobs=len(inventory),new_model_fits=860,query_checks=len(checks),rows=len(p),ablation_rows=len(ab),all_conditions_have_ten_seeds=True,training_parity=json.loads((B/'parity_check.json').read_text()),inputs={str(f.relative_to(R)):sha(f) for f in [old_history,old_ablation,B/'registered_plan.json']},script_sha256=sha(Path(__file__)))
(O/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2));code=O/'code';code.mkdir(exist_ok=True)
for f in [Path(__file__),R/'scripts/run_ten_seed_extension.py']:shutil.copy2(f,code/f.name)
shutil.copy2(B/'registered_plan.json',O/'experiment_plan.json')
with zipfile.ZipFile(R/'deliverables/MHAP_PK_ten_seed_results_20261001.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(O.rglob('*')):
  if f.is_file():z.write(f,Path(O.name)/f.relative_to(O))
print(s[s.k==2][['track','condition','sb_hrmse_mean','sb_hrmse_std']].to_string(index=False));print('COMPLETE',manifest)
