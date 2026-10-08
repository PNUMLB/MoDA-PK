"""Verify and aggregate molecular/identity fits from original query predictions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from summarize_training_scope_extension import drug_metrics,summarize
from mhap_pk.evaluation.quality_metrics import extended_metrics
B=R/'results/runs/MOLECULAR_IDENTITY/20261002';O=R/'results/summaries/molecular_identity_20261002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
KEY=['source_id','drug_id','participant_id','profile_id','observation_id']

def main():
 assert json.loads((B/'status.json').read_text())['status']=='completed'
 for path,h in json.loads((B/'code_manifest.json').read_text()).items():assert sha(R/path)==h,path
 O.mkdir(parents=True,exist_ok=True)
 ab=pd.read_parquet(R/'results/summaries/ten_seeds_20261001/ablation_predictions.parquet')
 controls=ab[ab.condition.isin(['drug_id','molecular'])].copy()
 records=[];inventory=[];fits={};row_checks=[]
 def load(run,condition,reference=None):
  man=json.loads((run/'run_manifest.json').read_text());assert man['status']=='completed'
  assert sha(run/'predictions.parquet')==man['prediction_sha256']
  sel=json.loads((run/'selection.json').read_text());assert sha(run/'selected.pt')==sel['checkpoint_sha256']
  p=pd.read_parquet(run/'predictions.parquet').rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'})
  if reference is not None:
   raw=p[(p.protocol=='first_k')&(p.k==2)].set_index(KEY).sort_index();ref=reference.set_index(KEY).sort_index()
   pd.testing.assert_frame_equal(raw[['y_true_log','y_pred_log']],ref[['y_true_log','y_pred_log']])
  art=json.loads((run/'artifacts.json').read_text());cal=json.loads((run/'population.json').read_text());cfg=json.loads((run/'config.resolved.json').read_text());assert art['train']==cal['fitting_indices']
  fold=man['case'];seed=man['seed'];fits[(condition,fold,seed)]=(art,cal,cfg)
  p=p[(p.protocol=='first_k')&p.k.isin([0,1,2,4])].copy();p['condition']=condition;p['track']='Known-drug' if fold=='known' else 'Joint drug–source';p['run_id']=str(run.relative_to(R));p['dataset_id']=p['split_id']=cfg['dataset_id'];records.append(p)
  inventory.append(dict(condition=condition,fold=fold,seed=seed,run_id=str(run.relative_to(R)),prediction_sha256=sha(run/'predictions.parquet'),config_sha256=sha(run/'config.resolved.json'),artifact_sha256=sha(run/'artifacts.json'),seconds=man['elapsed_seconds'],**sel))
 for rid,g in controls.groupby('run_id'):
  run=R/rid if '/' in rid else R/'results/runs/TRACEABLE/evaluation_20260925'/rid
  load(run,g.condition.iloc[0],g)
 for j in json.loads((B/'registered_plan.json').read_text())['jobs']:load(B/f"{j['fold']}__s{j['seed']}__molecular",'molecular_identity')
 for condition,fold,seed in fits:
  a,c,cfg=fits[(condition,fold,seed)];ref,cal,base=fits[('molecular',fold,seed)]
  assert a==ref,(condition,fold,seed,'artifacts')
  assert c['log_parameters']==cal['log_parameters']
  for key in ['epochs','learning_rate','weight_decay','validate_every','dataset_id','split_id','split_sha256']:assert cfg[key]==base[key],(key,condition,fold,seed)
 p=pd.concat(records,ignore_index=True);assert not p.duplicated(['condition','seed','fold','k']+KEY).any()
 assert np.isfinite(p[['y_true_log','y_pred_log']]).all().all() and np.isfinite(np.exp(p.y_pred_log)).all()
 for (fold,seed),g in p.groupby(['fold','seed']):
  ref=g[(g.condition=='molecular')&(g.k==2)].set_index(KEY)[['query_time','y_true_log']].sort_index()
  for (condition,k),h in g.groupby(['condition','k']):
   pd.testing.assert_frame_equal(ref,h.set_index(KEY)[['query_time','y_true_log']].sort_index())
   row_checks.append(dict(fold=fold,seed=seed,condition=condition,k=int(k),queries=len(h),same_queries=True))
 p.to_parquet(O/'canonical_predictions.parquet',index=False)
 inv=pd.DataFrame(inventory);inv.to_csv(O/'run_inventory.csv',index=False)
 cohort=pd.DataFrame([dict(track=t,condition=c,seed=s,k=k,**extended_metrics(g)) for (t,c,s,k),g in p.groupby(['track','condition','seed','k'])])
 metrics=[c for c in cohort if c not in ['track','condition','seed','k']];cs=summarize(cohort,['track','condition','k'],metrics)
 d,part=drug_metrics(p,['track','condition','seed','k','source_id','drug_id'])
 ds=summarize(d,['track','condition','source_id','drug_id','k','participants','queries'],['rmse','mae','twofold','bias','p90_abs_log_error'])
 extra=pd.DataFrame([dict(track=t,condition=c,seed=s,k=k,source_id=source,drug_id=drug,**extended_metrics(g)) for (t,c,s,k,source,drug),g in p.groupby(['track','condition','seed','k','source_id','drug_id'])])
 paired=[];pk=['track','seed','k','source_id','drug_id']
 for baseline in ['molecular','drug_id']:
  pairs=d[d.condition=='molecular_identity'].merge(d[d.condition==baseline],on=pk,suffixes=('','_reference'),validate='one_to_one')
  pairs['reference']=baseline
  for c in ['rmse','mae','twofold','bias']:pairs['delta_'+c]=pairs[c]-pairs[c+'_reference']
  paired.append(pairs)
 paired=pd.concat(paired,ignore_index=True);paireds=summarize(paired,['track','drug_id','k','reference'],['delta_rmse','delta_mae','delta_twofold','delta_bias'])
 # Same checkpoint, same future queries; k0 is a no-observation-update inference control.
 history=part[part.k==2].merge(part[part.k==0],on=['track','condition','seed','source_id','drug_id','participant_id'],suffixes=('','_k0'),validate='one_to_one')
 history['delta_mse']=history.mse-history.mse_k0;history['worsened']=history.delta_mse>1e-10
 hist=history.groupby(['track','condition','drug_id','seed']).agg(mean_delta_mse=('delta_mse','mean'),worsened_participant_fraction=('worsened','mean')).reset_index()
 # Bias/shape decomposition uses future residuals solely for retrospective diagnostics.
 x=p[p.k==2].copy();x['e']=x.y_pred_log-x.y_true_log;x['sq']=x.e.square() if hasattr(x.e,'square') else x.e**2
 bias=x.groupby(['track','condition','seed','source_id','drug_id','participant_id','profile_id']).agg(mse=('sq','mean'),bias=('e','mean'),queries=('e','size')).reset_index();bias['bias_squared']=bias.bias**2;bias['centered_mse']=(bias.mse-bias.bias_squared).clip(lower=0)
 frames={'cohort_by_seed':cohort,'cohort_summary':cs,'drug_by_seed':d,'drug_summary':ds,'drug_additional_metrics_by_seed':extra,'paired_drug_by_seed':paired,'paired_drug_summary':paireds,'history_effect_by_participant':history,'history_effect_by_drug_seed':hist,'profile_bias_shape':bias,'query_checks':pd.DataFrame(row_checks),'run_inventory':inv}
 for name,f in frames.items():f.to_csv(O/(name+'.csv'),index=False)
 with pd.ExcelWriter(O/'molecular_identity_results.xlsx') as w:
  for name,f in frames.items():f.to_excel(w,sheet_name=name[:31],index=False)
 main=cs[cs.k==2][['track','condition','sb_hrmse_mean','sb_hrmse_std','sb_log_mae_mean','sb_within_twofold_pct_mean']]
 text=['# Uni-Mol + Drug identity 입력 결합 검증','', '신규100 fits + 검증된 기존200 fits. 각 조건10 seeds, 동일분할·120epochs·relation OFF. Test 기반 재선택 없는 내부 탐색 비교.','',main.to_markdown(index=False),'','## 약물별 log-RMSE (k=2)','',ds[ds.k==2].pivot(index=['track','drug_id'],columns='condition',values='rmse_mean').to_markdown(),'','## 결합 모델의 단독 입력 대비 개선 약물 수','']
 for (track,ref),g in paireds[paireds.k==2].groupby(['track','reference']):text.append(f'- {track}, {ref} 대비: {int((g.delta_rmse_mean<0).sum())}/{len(g)}개. Seed 평균 차이이며 통계적 유의성 주장이 아님.')
 text+=['','## 모델 용량 및 선택','',inv.groupby('condition').parameters.agg(['min','max']).to_markdown(),'','결합은1024→32, 단독입력은512→32. ID 임베딩은 학습약물 수에 비례. 용량을 맞춘 순수정보효과 증명이 아니다. Unknown은 학습ID 평균이며 test별 ID 학습 없음. 관측효과는 동일 checkpoint의 k0→k2 비교이며 재학습 ablation과 다르다. Bias/shape 분해는 미래정답을 사용한 사후진단으로 보정모델의 성능이 아니다.','', '기존 주 모델은 관계손실ON이므로 이 표와 직접적인 구조원인 비교로 섞지 않는다. 기존 약물별5차원 residual 실험과 이번 입력연결은 다른 구조다.']
 (O/'REPORT_KO.md').write_text('\n'.join(text))
 (O/'verification.json').write_text(json.dumps(dict(status='completed',new_fits=100,reused_fits=200,seeds=list(range(1,11)),matching_fit_artifacts=300,query_checks=len(row_checks),prediction_rows=len(p),all_prediction_checksums_verified=True,exact_checkpoint_replay_in_new_runs=True,metric_implementation=sha(R/'src/mhap_pk/metrics.py')),indent=2))
 print(main.to_string(index=False))
if __name__=='__main__':main()
