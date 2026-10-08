"""Aggregate registered OOF and held-out predictions; never average fold RMSEs."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from mhap_pk.evaluation.drug_metrics import balanced_metrics
from run_resampling_validation import B,S,output
O=R/'results/summaries/resampling_20261002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=json.loads((B/'registered_plan.json').read_text());frames=[];inventory=[]
 for i,j in enumerate(plan['jobs']):
  path=output(j);manifest=json.loads((path/'run_manifest.json').read_text());assert manifest['status']=='completed'
  assert manifest['split_sha256']==plan['split_sha256']
  p=path/'predictions.parquet';assert sha(p)==manifest['prediction_sha256'];d=pd.read_parquet(p).rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'})
  d['evaluation_partition']=j['partition'];d['evaluation']='CV5' if j['partition'].startswith('cv5') else j['partition'];d['model']=j['label'];d['training_scope']=j['scope'];d['run_path']=str(path.relative_to(R));d['split_id']=S.name
  if j['kind']=='poppk':
   for scope in ['pooled','single']:x=d.copy();x['training_scope']=scope;frames.append(x)
  else:frames.append(d)
  inventory.append(dict(job=i,partition=j['partition'],scope=j['scope'],model=j['label'],seed=j['seed'],path=str(path.relative_to(R)),prediction_sha256=sha(p),rows=len(d)))
 d=pd.concat(frames,ignore_index=True);full=d[(d.training_scope=='pooled')&(d.model=='molecular_relation')].copy();full['training_scope']='ablation';d=pd.concat([d,full],ignore_index=True)
 idcols=['evaluation','training_scope','model','seed','protocol','k','source_id','drug_id','participant_id','profile_id','observation_id']
 assert not d.duplicated(idcols).any();assert np.isfinite(d[['y_true_log','y_pred_log']]).all().all();assert np.isfinite(np.exp(d[['y_true_log','y_pred_log']])).all().all()
 checks=[]
 for key,g in d.groupby(['evaluation','training_scope','seed','protocol','k'],sort=False):
  ref=None
  for model,h in g.groupby('model'):
   a=h.set_index(['source_id','drug_id','participant_id','profile_id','observation_id']).y_true_log.sort_index()
   if ref is None:ref=a
   else:assert ref.equals(a),(key,model)
  checks.append(dict(zip(['evaluation','training_scope','seed','protocol','k'],key))|dict(query_count=len(ref),matched_models=g.model.nunique()))
 # CV profiles must cover every eligible participant once, not one score per fold.
 from mhap_pk.data.traceable import load_profiles
 ps,meta,_=load_profiles();expected=set(str(q) for p in ps for q in p['query'].observation_id)
 for key,g in d[d.evaluation=='CV5'].groupby(['training_scope','model','seed','protocol','k']):assert set(g.observation_id.astype(str))==expected,key
 O.mkdir(exist_ok=False,parents=True);d.to_parquet(O/'canonical_predictions.parquet',index=False);pd.DataFrame(inventory).to_csv(O/'run_inventory.csv',index=False);pd.DataFrame(checks).to_csv(O/'query_audit.csv',index=False)
 def calculate(p):
  rows=[];keys=['evaluation','training_scope','model','seed','protocol','k','source_id','drug_id']
  for key,g in p.groupby(keys,sort=False):
   people=g.source_id.astype(str)+'::'+g.participant_id.astype(str);rows.append(dict(zip(keys,key))|balanced_metrics(g.y_true_log,g.y_pred_log,people))
  return pd.DataFrame(rows)
 metric=calculate(d);metric.to_csv(O/'drug_metrics_by_seed.csv',index=False)
 keys=['evaluation','training_scope','model','protocol','k','source_id','drug_id'];fields=['log_rmse','log_mae','within_twofold','signed_log_bias','linear_rmse','r2_log','r2_linear']
 su=metric.groupby(keys)[fields].agg(['mean','std','count']);su.columns=['_'.join(x) for x in su.columns];su=su.reset_index();su.to_csv(O/'drug_metrics_summary.csv',index=False)
 scores=metric.groupby(['evaluation','training_scope','model','seed','protocol','k','source_id']).log_rmse.mean().groupby(['evaluation','training_scope','model','seed','protocol','k']).mean().rename('sb_hrmse').reset_index();scores.to_csv(O/'hierarchical_scores_by_seed.csv',index=False)
 # Matched test subset helps distinguish training fraction effects from test-population changes.
 small=d[d.evaluation=='ratio_811'];small_ids=set(small.observation_id);matched=d[d.evaluation.isin(['ratio_712','ratio_811'])&d.observation_id.isin(small_ids)].copy();mm=calculate(matched);mm.to_csv(O/'nested_test_drug_metrics.csv',index=False)
 with pd.ExcelWriter(O/'resampling_results.xlsx') as w:
  su.to_excel(w,sheet_name='drug_summary',index=False);metric.to_excel(w,sheet_name='drug_seed_metrics',index=False);scores.to_excel(w,sheet_name='secondary_summary',index=False);mm.to_excel(w,sheet_name='matched_test_subset',index=False);pd.read_csv(S/'actual_counts.csv').to_excel(w,sheet_name='actual_split_counts',index=False)
 lines=['# 참가자 교차검증 및 분할 비율 평가','', '새로 학습한 결과입니다. 기존 6:2:2와 다른 split이며 기존 표를 덮어쓰지 않습니다. 이미 개발에 사용한 코호트의 내부 검증입니다.','', 'CV5는 참가자별 미학습 OOF 예측을 합친 뒤 약물별 지표를 계산합니다. fold 점수의 단순 평균이 아닙니다. 10개 seed는 학습 반복이며 50개 독립 평가가 아닙니다.','', '7:1:2와 8:1:1의 전체 test 구성은 다릅니다. 동일 validation 및 중첩된 test subset에서의 추가 비교를 제공합니다. 약물별 실제 인원은 actual_split_counts 시트 참조.','']
 for (evaluation,scope),g in su[(su.protocol=='first_k')&(su.k==2)].groupby(['evaluation','training_scope']):
  lines += [f'## {evaluation} / {scope}',g.pivot(index='drug_id',columns='model',values='log_rmse_mean').to_markdown(floatfmt='.4f'),'']
 (O/'REPORT_KO.md').write_text('\n'.join(lines));manifest=dict(status='completed',plan_sha256=sha(B/'registered_plan.json'),split_sha256=sha(S/'partitions.json'),jobs=len(inventory),fits=plan['fit_count'],query_alignment_passed=True,oof_full_coverage=True,aggregation='Out-of-fold queries combined before participant-balanced drug metrics, one score per seed; no fold-RMSE averaging',files={p.name:sha(p) for p in O.iterdir() if p.is_file()});(O/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2));print('SUMMARY COMPLETED',O,flush=True)
if __name__=='__main__':main()
