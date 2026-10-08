"""Rebuild training-scope results from immutable query predictions, never score joins."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.quality_metrics import extended_metrics
B=R/'results/runs/TRAINING_SCOPE/20261001';O=R/'results/summaries/training_scope_extension_20261001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
N={'single_drug':'MoDE-PK (Ours)','poppk_vi_map':'PopPK (MAP)','dcm':'DCM','unipk':'Uni-PK','deepct':'DeepCt','target_molecular':'Target only','pooled_molecular':'Pooled, no relations','pooled_molecular_relation':'Pooled MHAP-PK','pooled_no_molecule':'Pooled, no molecule','pooled_drug_id':'Pooled, drug identity'}
pretty=lambda d:'R-lipoic acid' if d=='r_lipoic_acid' else d.replace('_',' ').capitalize()
METRICS=['sb_hrmse','log_rmse','log_mae','within_twofold_pct','signed_bias','p90_abs_log_error','sb_log_mae','sb_within_twofold_pct']
def summarize(f,by,metrics):
 z=f.groupby(by).agg({c:['mean','std'] for c in metrics});z.columns=['_'.join(c) for c in z.columns];return z.reset_index()
def drug_metrics(p,group):
 p=p.copy();p['sq']=(p.y_pred_log-p.y_true_log)**2;p['ae']=abs(p.y_pred_log-p.y_true_log);p['twofold']=100*(p.ae<=np.log(2));p['bias']=p.y_pred_log-p.y_true_log
 part=p.groupby(group+['participant_id']).agg(mse=('sq','mean'),mae=('ae','mean'),twofold=('twofold','mean'),bias=('bias','mean'),queries=('observation_id','size')).reset_index()
 d=part.groupby(group).agg(mse=('mse','mean'),mae=('mae','mean'),twofold=('twofold','mean'),bias=('bias','mean'),participants=('participant_id','size'),queries=('queries','sum')).reset_index();d['rmse']=np.sqrt(d.mse)
 tail=p.groupby(group).ae.quantile(.9).rename('p90_abs_log_error').reset_index();return d.merge(tail,on=group),part

def main():
 assert json.loads((B/'status.json').read_text())['status']=='completed'
 for item in json.loads((B/'input_manifest.json').read_text())['files']:assert sha(R/item['path'])==item['sha256']
 for path,expected in json.loads((B/'code_manifest.json').read_text()).items():assert sha(R/path)==expected,path
 O.mkdir(parents=True,exist_ok=True);plan=json.loads((B/'registered_plan.json').read_text())
 old=R/'results/summaries/ten_seeds_20261001/canonical_predictions.parquet'
 existing=pd.read_parquet(old,filters=[('track','==','Known-drug')]);existing['run_id']=existing.run_id.map(lambda x:'results/runs/TRACEABLE_POPPK/evaluation_20260925/'+x if x.startswith('full__s') else x);single=[existing[existing.condition.isin(['single_drug','poppk_vi_map'])].copy()];low=[];inventory=[];selections=[]
 for j in plan['jobs']:
  root=R/j['root'];modes=[j['mode']] if j['kind']=='single' else j['modes']
  for mode in modes:
   run=root/f"known__s{j['seed']}__{mode}";path=run/'predictions.parquet';manifest=json.loads((run/'run_manifest.json').read_text());assert manifest['status']=='completed' and sha(path)==manifest['prediction_sha256']
   p=pd.read_parquet(path,filters=[('protocol','==','first_k')]).rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'})
   assert set(p.seed)=={j['seed']} and set(p.k)==set(range(5)) and set(p.drug_id)=={j['drug']}
   assert np.isfinite(p[['y_true_log','y_pred_log']]).all().all() and np.isfinite(np.exp(p.y_pred_log)).all()
   p['run_id']=str(run.relative_to(R));p['dataset_id']=p['split_id']='source_traceable_20260925'
   if j['kind']=='single':single.append(p)
   else:
    p['condition']=j['scope']+'_'+mode;p['target_n']=j['cap'];p['full_target_n']=json.loads((root/'plan.json').read_text())['full_target_training_participants'];low.append(p)
   a=json.loads((run/'artifacts.json').read_text());cal=json.loads((run/'population.json').read_text());assert a['train']==cal['fitting_indices']
   inventory.append(dict(run_id=str(run.relative_to(R)),kind=j['kind'],drug_id=j['drug'],seed=j['seed'],condition=p.condition.iloc[0],prediction_sha256=sha(path),checkpoint_sha256=sha(run/'selected.pt'),config_sha256=sha(run/'config.resolved.json'),calibration_sha256=sha(run/'population.json'),training_partition_sha256=hashlib.sha256(json.dumps(a['train']).encode()).hexdigest(),seconds=manifest['elapsed_seconds'],query_rows=len(p)))
   sel=json.loads((run/'selection.json').read_text());selections.append(inventory[-1]|sel)
 s=pd.concat(single,ignore_index=True);l=pd.concat(low,ignore_index=True)
 # Existing full-data target-only fits: exact model/validation/calibration recipe, no retraining.
 counts=pd.read_csv(R/'results/summaries/single_drug_20261001/partition_counts.csv')
 full=s[s.condition=='single_drug'].copy().merge(counts[['drug_id','train_participants']],on='drug_id',validate='many_to_one')
 full['condition']='target_molecular';full['target_n']=full['full_target_n']=full.train_participants;l=pd.concat([l,full],ignore_index=True)
 reused=[]
 for rid,g in existing[existing.condition.isin(['single_drug','poppk_vi_map'])].groupby('run_id'):
  rp=R/rid;raw=rp/'predictions.parquet';rm=json.loads((rp/'run_manifest.json').read_text());assert rm['status']=='completed'
  h=sha(raw);assert h==rm['prediction_sha256']
  rawp=pd.read_parquet(raw).rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});rawp=rawp[rawp.protocol=='first_k'] if 'protocol' in rawp else rawp
  kk=['seed','k','observation_id'];rawp=rawp.set_index(kk);gg=g.set_index(kk);assert rawp.index.is_unique and gg.index.is_unique
  assert np.array_equal(rawp.loc[gg.index,['y_true_log','y_pred_log']].to_numpy(),gg[['y_true_log','y_pred_log']].to_numpy()),rid
  reused.append(dict(run_id=rid,condition=g.condition.iloc[0],seed=int(g.seed.iloc[0]),prediction_sha256=h,canonical_prediction_parent=str(old.relative_to(R)),additional_inference_k3='Included in parent canonical prediction manifest; original fit unchanged'))
  if g.condition.iloc[0]=='single_drug' and (rp/'selection.json').exists():
   selections.append(dict(run_id=rid,kind='single_reused',drug_id=g.drug_id.iloc[0],seed=int(g.seed.iloc[0]),condition='single_drug',**json.loads((rp/'selection.json').read_text())))
 pd.DataFrame(reused).to_csv(O/'reused_run_inventory.csv',index=False)
 checks=[];idcols=['source_id','drug_id','participant_id','profile_id','query_time','y_true_log']
 for (drug,seed),g in s.groupby(['drug_id','seed']):
  ref=g[(g.condition=='single_drug')&(g.k==2)].set_index('observation_id').sort_index()
  for (mode,k),a in g.groupby(['condition','k']):
   a=a.set_index('observation_id').sort_index();assert a.index.is_unique and a.index.equals(ref.index) and np.array_equal(a[idcols],ref[idcols]);checks.append(dict(kind='single',drug=drug,seed=seed,condition=mode,k=k,rows=len(a),passed=True))
  h=l[(l.drug_id==drug)&(l.seed==seed)]
  for (mode,n,k),a in h.groupby(['condition','target_n','k']):
   a=a.set_index('observation_id').sort_index();assert a.index.is_unique and a.index.equals(ref.index) and np.array_equal(a[idcols],ref[idcols]);checks.append(dict(kind='low_data',drug=drug,seed=seed,condition=mode,n=n,k=k,rows=len(a),passed=True))
 for p,groups in [(s,['condition','drug_id','k']),(l,['condition','drug_id','target_n','k'])]:
  for _,g in p.groupby(groups):assert set(g.seed)==set(range(1,11))
 s.to_parquet(O/'single_drug_predictions.parquet',index=False);l.to_parquet(O/'low_data_predictions.parquet',index=False)
 sd,sp=drug_metrics(s,['condition','seed','k','source_id','drug_id']);ld,lp=drug_metrics(l,['condition','seed','k','source_id','drug_id','target_n','full_target_n'])
 ds=['rmse','mae','twofold','bias','p90_abs_log_error']
 ss=summarize(sd,['condition','drug_id','k','participants','queries'],ds).merge(counts,on='drug_id',validate='many_to_one');ls=summarize(ld,['condition','drug_id','target_n','full_target_n','k','participants','queries'],ds)
 cohort=pd.DataFrame([dict(condition=c,seed=seed,k=k,**extended_metrics(g)) for (c,seed,k),g in s.groupby(['condition','seed','k'])]);cs=summarize(cohort,['condition','k'],METRICS)
 key=['seed','k','source_id','drug_id','target_n','full_target_n'];ref=ld[ld.condition=='target_molecular'][key+ds];pairs=ld[ld.condition!='target_molecular'].merge(ref,on=key,suffixes=('','_target'),validate='many_to_one')
 for c in ds:pairs['delta_'+c]=pairs[c]-pairs[c+'_target']
 ps=summarize(pairs,['condition','drug_id','target_n','full_target_n','k'],['delta_'+c for c in ds])
 # Negative transfer at identical participant/query sets, keeping repeated profiles together.
 pk=key+['participant_id'];pr=lp[lp.condition=='target_molecular'][pk+['mse']];pp=lp[lp.condition!='target_molecular'].merge(pr,on=pk,suffixes=('','_target'),validate='many_to_one');pp['delta_mse']=pp.mse-pp.mse_target;pp['worsened']=pp.delta_mse>1e-10
 adverse=pp.groupby(['condition','drug_id','target_n','seed','k']).agg(mean_delta_mse=('delta_mse','mean'),worsened_participant_fraction=('worsened','mean'),max_delta_mse=('delta_mse','max')).reset_index()
 frames={'single_drug_metrics_by_seed':sd,'single_drug_summary':ss,'single_drug_cohort_by_seed':cohort,'single_drug_cohort_summary':cs,'low_data_metrics_by_seed':ld,'low_data_summary':ls,'low_data_paired_by_seed':pairs,'low_data_paired_summary':ps,'low_data_adverse_transfer':adverse,'run_inventory':pd.DataFrame(inventory),'selection_audit':pd.DataFrame(selections),'query_checks':pd.DataFrame(checks),'participant_counts':counts}
 for name,f in frames.items():f.to_csv(O/(name+'.csv'),index=False)
 primary=[];low_primary=[]
 for drug,g in ss[ss.k==2].groupby('drug_id'):
  g=g.set_index('condition');a=g.iloc[0];row=dict(Drug=pretty(drug),Training_participants=int(a.train_participants),Validation_participants=int(a.validation_participants),Test_participants=int(a.participants),Future_measurements=int(a.queries))
  for c in ['single_drug','poppk_vi_map','dcm','unipk','deepct']:row[N[c]]=f'{g.loc[c,"rmse_mean"]:.4f} ± {g.loc[c,"rmse_std"]:.4f}'
  primary.append(row)
 for drug,g in ls[(ls.k==2)&(ls.target_n==2)].groupby('drug_id'):
  g=g.set_index('condition');row=dict(Drug=pretty(drug),Target_training_participants=2)
  for c in ['target_molecular','pooled_molecular','pooled_molecular_relation']:row[N[c]]=f'{g.loc[c,"rmse_mean"]:.4f} ± {g.loc[c,"rmse_std"]:.4f}'
  low_primary.append(row)
 pd.DataFrame(primary).to_csv(O/'single_drug_primary_table.csv',index=False);pd.DataFrame(low_primary).to_csv(O/'low_data_primary_table.csv',index=False)
 with pd.ExcelWriter(O/'training_scope_results.xlsx',engine='openpyxl') as w:
  pd.DataFrame(primary).to_excel(w,sheet_name='Main single-drug table',index=False);pd.DataFrame(low_primary).to_excel(w,sheet_name='Low-data n2 table',index=False)
  for name,f in frames.items():f.to_excel(w,sheet_name=name[:31],index=False)
  for ws in w.book.worksheets:
   ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
   for col in ws.columns:
    ws.column_dimensions[col[0].column_letter].width=min(42,max(14,len(str(col[0].value))+2))
 mods=['single_drug','poppk_vi_map','dcm','unipk','deepct'];rows=['# 개별 약물 비교와 저자료량 통합 학습','', '모든 수치는 query 예측에서 재집계했다. Seeds 1–10, 주 조건 k=2. 학습 epoch/validation budget은 기존과 같으며 선택에 test 오차를 쓰지 않았다.','', '## 약물별 개별 학습 비교','', '| Drug | Train / validation / test people | '+' | '.join(N[c] for c in mods)+' |','|---|---|'+'---:|'*len(mods)]
 wins=0
 for drug,g in ss[ss.k==2].groupby('drug_id'):
  g=g.set_index('condition');best=g.loc[mods].rmse_mean.min();wins+=int(g.loc['single_drug','rmse_mean']==best);values=[]
  for c in mods:
   a=g.loc[c];v=f'{a.rmse_mean:.4f} ± {a.rmse_std:.4f}';values.append('**'+v+'**' if a.rmse_mean==best else v)
  a=g.iloc[0];rows.append(f'| {pretty(drug)} | {int(a.train_participants)} / {int(a.validation_participants)} / {int(a.participants)} | '+' | '.join(values)+' |')
 rows+=['',f'MHAP-PK의 평균 participant-balanced drug log-RMSE가 가장 낮은 약물: {wins}/11. 유의성 검정 결과가 아니다. 단일 약물에서는 분자 표현이 상수이므로 이 비교는 분자 전이의 효과를 입증하지 않는다.','', '## 저자료량 통합 학습: 약물별 paired 변화','', '아래는 pooled − target-only log-RMSE. 음수는 통합 학습 개선, 양수는 악화. 같은 n에서도 약물별 validation 참가자 수는 고정되어 다르다. Seeds는 subset 선택과 초기화를 함께 바꾼다.','', '| Drug | n=2 no relations | n=2 full MHAP-PK | Full-n no relations | Full-n full MHAP-PK |','|---|---:|---:|---:|---:|']
 for drug,g in ps[ps.k==2].groupby('drug_id'):
  vals=[]
  for full,cond in [(False,'pooled_molecular'),(False,'pooled_molecular_relation'),(True,'pooled_molecular'),(True,'pooled_molecular_relation')]:
   a=g[(g.condition==cond)&(g.target_n==g.full_target_n if full else g.target_n==2)].iloc[0];vals.append(f'{a.delta_rmse_mean:+.4f} ± {a.delta_rmse_std:.4f}')
  rows.append('| '+pretty(drug)+' | '+' | '.join(vals)+' |')
 rows+=['','Co-administration links are retained: all drug records of omitted target participants are removed from donor training. Every reduced fit recalibrates its population initialization and molecular scaler on that exact training partition. Relation-free target/pooled comparison separates donor availability from the added relation objective, but still changes optimization sample composition. All k=0–4, drugs, and negative effects are retained in Excel/CSV. Molecular ablation at n=2 distinguishes identity/no-molecule controls; no external validation is claimed.']
 (O/'REPORT_KO.md').write_text('\n'.join(rows))
 manifest=dict(status='completed',new_fits=len(inventory),seeds=list(range(1,11)),primary_k=2,single_drug_wins=wins,query_checks=len(checks),query_checks_passed=True,input_manifest_sha256=sha(B/'input_manifest.json'),registered_plan_sha256=sha(B/'registered_plan.json'),source_snapshot_sha256=sha(B/'source_snapshot.tar.gz'),reused_predictions_sha256=sha(old),script_sha256=sha(Path(__file__)),outputs={f.name:sha(f) for f in O.iterdir() if f.is_file() and f.name!='analysis_manifest.json'})
 (O/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='outputs'},indent=2))
if __name__=='__main__':main()
