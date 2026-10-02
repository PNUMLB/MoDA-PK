"""Recompute the encoder comparison exclusively from matched query predictions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from summarize_training_scope_extension import drug_metrics,summarize
from mhap_pk.evaluation.quality_metrics import extended_metrics
B=R/'results/runs/MOLECULAR_ENCODER/20261002';O=R/'results/summaries/molecular_encoder_20261002'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
PANEL=['no_molecule','drug_id','ecfp','unimol','chemberta_mlm','chemberta_mtr']
N=dict(no_molecule='Without molecular features',drug_id='Drug identity',ecfp='ECFP',unimol='Uni-Mol',chemberta_mlm='ChemBERTa MLM',chemberta_mtr='ChemBERTa MTR',full_mhap='MHAP-PK (both relations)',target_only='Target-only (no relations)')
def interpretation(cs,ds,paired):
 c=cs[cs.k==2].set_index(['track','condition']);known=lambda m:c.loc[('Known-drug',m),'sb_hrmse_mean'];joint=lambda m:c.loc[('Joint drug–source',m),'sb_hrmse_mean']
 change=100*(known('unimol')-known('chemberta_mlm'))/known('unimol')
 w=ds[(ds.track=='Joint drug–source')&(ds.k==2)&(ds.drug_id=='warfarin')].set_index('condition')
 return ['## 핵심 발견',f"ChemBERTa MLM의 Known SB-hRMSE는 {known('chemberta_mlm'):.4f}로 Uni-Mol {known('unimol'):.4f}보다 {change:.2f}% 낮다. 그러나 Joint는 {joint('chemberta_mlm'):.4f}로 Uni-Mol {joint('unimol'):.4f}보다 높다. 두 조건 모두 개선한 표현 교체는 아니다.",
 f"ChemBERTa MTR의 Joint 오류는 {joint('chemberta_mtr'):.4f}다. Warfarin 약물별 log-RMSE는 Uni-Mol {w.loc['unimol','rmse_mean']:.4f}에서 MTR {w.loc['chemberta_mtr','rmse_mean']:.4f}로 증가했고 signed log bias는 {w.loc['chemberta_mtr','bias_mean']:.4f}로 큰 과소예측을 보였다. 모든 약물이 나빠진 결과로 해석하지 않는다.",
 'MLM은 Uni-Mol보다 Known/Joint/n=2 각각6/11개 약물에서 낮은 평균오차를 보였다. 그러나 target-only n=2와 비교하면3/11개에서만 개선했다(Uni-Mol pooling은5/11개). 평균 pooled 개선이 모든 약물의 transfer 개선을 의미하지 않는다.',
 f"Joint에서는 Drug-ID fallback {joint('drug_id'):.4f}와 no-molecule {joint('no_molecule'):.4f}가 네 분자표현보다 낮은 SB-hRMSE를 보인다. 이 cohort에서 molecular transfer가 unknown 조건의 우월성을 확보했다는 주장은 지지되지 않는다.",
 'Known MTR은10/10seeds에서120epoch 상한을 선택했다. 이번 결과는 동일 고정 예산에서의 비교이며 encoder별 충분한 최적화가 끝난 성능 상한으로 해석하지 않는다. Validation budget 확장은 별도 사전등록된 후속 비교로 다룬다.',
 '기존 full MHAP-PK를 그대로 유지하고, MLM의 Known 개선을 후보 근거로 기록한다. Chemical/PK relation과의 결합 효과는 현재 결과로 추정하지 않는다.']

def main():
 assert json.loads((B/'status.json').read_text())['status']=='completed'
 for f,h in json.loads((B/'code_manifest.json').read_text()).items():assert sha(R/f)==h,f
 for x in json.loads((B/'input_manifest.json').read_text())['files']:assert sha(R/x['path'])==x['sha256']
 assert json.loads((B/'parity/verification.json').read_text())['exact_original_predictions']
 O.mkdir(parents=True,exist_ok=True);rows=[];inventory=[]
 for j in json.loads((B/'registered_plan.json').read_text())['jobs']:
  root=R/j['root'];run=root if j['kind']=='inference' else root/f"{j['fold']}__s{j['seed']}__molecular"
  m=json.loads((run/'run_manifest.json').read_text());assert m['status']=='completed';assert sha(run/'predictions.parquet')==m['prediction_sha256']
  p=pd.read_parquet(run/'predictions.parquet').rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});p=p[p.protocol=='first_k'].copy();p['condition']=j['encoder'];p['track']='Low-data n=2' if j['scope']=='low_data' else ('Known-drug' if j['fold']=='known' else 'Joint drug–source');p['run_id']=str(run.relative_to(R));rows.append(p)
  sel={}
  if j['kind']=='train':
   sel=json.loads((run/'selection.json').read_text());assert sel['checkpoint_sha256']==sha(run/'selected.pt')
   art=json.loads((run/'artifacts.json').read_text());cal=json.loads((run/'population.json').read_text());assert art['train']==cal['fitting_indices']
   if j['scope']=='panel':
    parent=R/('results/runs/TRACEABLE/evaluation_20260925' if j['seed']<=5 else 'results/runs/TEN_SEEDS/20261001/panel')/f"{j['fold']}__s{j['seed']}__molecular"
    original=json.loads((parent/'artifacts.json').read_text())
   else:original=json.loads((root/'partition.json').read_text())
   for role in ['train','validation','test']:assert art[role]==original[role],(j,role)
   assert set(art['train']).isdisjoint(art['test']) and set(art['validation']).isdisjoint(art['test'])
   sc=art['scaler'];assert sc['representation']==j['encoder'] if j['encoder'].startswith('chemberta') else True
   assert json.loads((run/'verification.json').read_text())['max_prediction_difference']<1e-10
  else:
   assert m['exact_parent_predictions'] and m['parent_checkpoint_sha256']==sha(R/j['parent']/'selected.pt')
   sel=json.loads((R/j['parent']/'selection.json').read_text())
  inventory.append(dict(run_id=str(run.relative_to(R)),kind=j['kind'],track=p.track.iloc[0],condition=j['encoder'],fold=j['fold'],seed=j['seed'],drug=j.get('drug'),prediction_sha256=sha(run/'predictions.parquet'),elapsed_seconds=m.get('elapsed_seconds'),**sel))
 previous=R/'results/summaries/half_head_20261001/canonical_predictions.parquet';old=pd.read_parquet(previous);old=old[old.condition.isin(['shared','full_mhap','target_only'])].copy();old['condition']=old.condition.replace({'shared':'unimol'});rows.append(old)
 previous2=R/'results/summaries/training_scope_extension_20261001/low_data_predictions.parquet';low=pd.read_parquet(previous2);low=low[(low.target_n.astype(str)=='2')&low.condition.isin(['pooled_no_molecule','pooled_drug_id'])].copy();low['condition']=low.condition.replace({'pooled_no_molecule':'no_molecule','pooled_drug_id':'drug_id'});low['track']='Low-data n=2';rows.append(low)
 reused=[]
 for rid,g in pd.concat([old,low]).groupby('run_id'):
  run=R/rid;raw=run/'predictions.parquet';m=json.loads((run/'run_manifest.json').read_text());assert m['status']=='completed' and sha(raw)==m['prediction_sha256']
  r=pd.read_parquet(raw).rename(columns={'y_true_lnc':'y_true_log','y_pred_lnc':'y_pred_log'});r=r[r.protocol=='first_k'];ix=['seed','k','observation_id'];r=r.set_index(ix);g=g.set_index(ix);assert r.index.is_unique and np.array_equal(r.loc[g.index,['y_true_log','y_pred_log']],g[['y_true_log','y_pred_log']]),rid
  reused.append(dict(run_id=rid,prediction_sha256=sha(raw),rows=len(g)))
 p=pd.concat(rows,ignore_index=True);p['dataset_id']=p['split_id']='source_traceable_20260925';assert np.isfinite(p[['y_true_log','y_pred_log']]).all().all() and np.isfinite(np.exp(p.y_pred_log)).all()
 assert not p.duplicated(['track','condition','seed','k','observation_id']).any()
 checks=[];cols=['source_id','drug_id','participant_id','profile_id','query_time','y_true_log']
 for (track,seed,drug),g in p.groupby(['track','seed','drug_id']):
  ref=g[(g.condition=='unimol')&(g.k==2)].set_index('observation_id').sort_index();assert len(ref)
  assert set(PANEL)<=set(g.condition),(track,seed,drug,set(g.condition))
  for (cond,k),a in g.groupby(['condition','k']):
   a=a.set_index('observation_id').sort_index();assert a.index.equals(ref.index) and np.array_equal(a[cols],ref[cols]),(track,seed,drug,cond,k)
   if k:assert (a.query_time>a.support_end_time).all()
   checks.append(dict(track=track,seed=seed,drug=drug,condition=cond,k=k,rows=len(a),passed=True))
 for key,g in p.groupby(['track','condition','drug_id','k']):assert set(g.seed)==set(range(1,11)),key
 by=['track','condition','seed','k','source_id','drug_id'];d,part=drug_metrics(p,by);ds=summarize(d,[x for x in by if x!='seed'],['rmse','mae','twofold','bias','p90_abs_log_error','participants','queries'])
 c=pd.DataFrame([dict(track=t,condition=m,seed=s,k=k,**extended_metrics(g)) for (t,m,s,k),g in p.groupby(['track','condition','seed','k'])]);cs=summarize(c,['track','condition','k'],['sb_hrmse','log_rmse','log_mae','within_twofold_pct','sb_log_mae','sb_within_twofold_pct','signed_bias','p90_abs_log_error'])
 pairs=[];harms=[]
 for reference in ['unimol','target_only']:
  ix=['track','seed','k','source_id','drug_id'];ref=d[d.condition==reference][ix+['rmse']].rename(columns={'rmse':'reference_rmse'});q=d.merge(ref,on=ix,validate='many_to_one');q['delta_rmse']=q.rmse-q.reference_rmse;q['reference']=reference;pairs.append(q)
  ix+=['participant_id'];ref=part[part.condition==reference][ix+['mse']].rename(columns={'mse':'reference_mse'});h=part.merge(ref,on=ix,validate='many_to_one');h['delta_mse']=h.mse-h.reference_mse;h['worse']=h.delta_mse>1e-10;h['reference']=reference;harms.append(h)
 pair=pd.concat(pairs,ignore_index=True);paired=summarize(pair,['track','condition','k','source_id','drug_id','reference'],['delta_rmse']);harm=pd.concat(harms,ignore_index=True).groupby(['track','condition','k','drug_id','seed','reference']).agg(worsened_participant_fraction=('worse','mean'),mean_delta_mse=('delta_mse','mean')).reset_index()
 inv=pd.DataFrame(inventory);select=inv[inv.kind=='train'].groupby(['track','condition']).agg(fits=('run_id','size'),epoch0=('epoch',lambda x:int((x==0).sum())),epoch120=('epoch',lambda x:int((x==120).sum())),median_epoch=('epoch','median'),median_seconds=('elapsed_seconds','median')).reset_index()
 features=[]
 for name,dim in [('ecfp',512),('unimol',512),('chemberta_mlm',384),('chemberta_mtr',384)]:
  row=dict(representation=name,native_dimension=dim,backbone_input_dimension=512,allocated_input_weights=512*32,nonpadding_input_weights=dim*32,encoder_trainable_parameters=0)
  if name.startswith('chemberta'):
   meta=json.loads((R/'data/features/molecular_comparison_20261002'/f'{name}.json').read_text());row.update(checkpoint=meta['repo'],revision=meta['revision'],encoder_parameters=meta['encoder_parameters'],embedding_sha256=meta['embedding_sha256'],unknown_tokens=sum(x['unknown_tokens'] for x in meta['audit']),batch_single_max_difference=meta['batch_single_max_difference'])
  features.append(row)
 outputs=dict(drug_by_seed=d,drug_summary=ds,cohort_by_seed=c,cohort_summary=cs,paired_by_seed=pair,paired_summary=paired,participant_adverse_transfer=harm,query_checks=pd.DataFrame(checks),new_run_inventory=inv,reused_run_inventory=pd.DataFrame(reused),selection_summary=select,feature_audit=pd.DataFrame(features))
 for name,f in outputs.items():f.to_csv(O/(name+'.csv'),index=False)
 p.to_parquet(O/'canonical_predictions.parquet',index=False)
 (O/'representation_registry.json').write_text(json.dumps({key:dict(display_name=value,role='Representation-only control inside MHAP-PK' if key in PANEL else 'Contextual comparison',relations='OFF' if key!='full_mhap' else 'Chemical and PK ON') for key,value in N.items()},indent=2))
 from openpyxl.styles import Font
 with pd.ExcelWriter(O/'molecular_encoder_results.xlsx',engine='openpyxl') as w:
  for name,f in outputs.items():f.to_excel(w,sheet_name=name[:31],index=False);w.sheets[name[:31]].freeze_panes='A2';w.sheets[name[:31]].auto_filter.ref=w.sheets[name[:31]].dimensions
 primary=ds[ds.k==2].pivot(index=['track','drug_id'],columns='condition',values='rmse_mean');primary.to_csv(O/'primary_drug_table.csv')
 with pd.ExcelWriter(O/'molecular_encoder_results.xlsx',engine='openpyxl',mode='a') as w:
  for ti,track in enumerate(['Known-drug','Joint drug–source','Low-data n=2']):
   for metric in ['rmse','mae','twofold']:
    a=ds[(ds.track==track)&(ds.k==2)&ds.condition.isin(PANEL)].pivot(index='drug_id',columns='condition',values=metric+'_mean').reindex(columns=PANEL).rename(columns=N)
    name=['Known','Joint','Low_n2'][ti]+'_'+metric;a.to_excel(w,sheet_name=name);sheet=w.sheets[name];sheet.freeze_panes='B2';sheet.column_dimensions['A'].width=24
    for col in range(2,8):sheet.column_dimensions[chr(64+col)].width=24
    for row in sheet.iter_rows(min_row=2):
     values=[x.value for x in row[1:]];best=(max if metric=='twofold' else min)(values)
     for cell in row[1:]:cell.number_format='0.0000' if metric!='twofold' else '0.00';cell.font=Font(bold=abs(cell.value-best)<1e-12)

 report=['# 분자 표현 비교: 관계 손실 OFF','10 seeds; 신규530fits + 기존checkpoint300건 재추론. Known-drug / Joint drug–source / 약물당 학습참가자2명 조건. k=0,1,2,3,4 중 주 설정k=2. 약물당 학습참가자 수 n과 개인에게 공개하는 관측 수k는 다르다.','자료: 출처 추적 가능한 기존11약물/9sources/482profiles. 같은 source 내 participant linkage와 기존 분할을 유지한다. 약물별 표는 참가자 균형 log-RMSE/MAE/twofold를 제공하며 P90 absolute log error는 query 가중치다.', '## 전체 요약',cs[cs.k==2][['track','condition','sb_hrmse_mean','sb_hrmse_std','sb_log_mae_mean','sb_within_twofold_pct_mean']].assign(condition=lambda x:x.condition.map(N)).to_markdown(index=False),'## 약물별 participant-balanced log-RMSE',primary.rename(columns=N).to_markdown(),'## Uni-Mol 대비 낮은 평균 오차를 보인 약물 수']
 for (t,m),g in paired[(paired.k==2)&(paired.reference=='unimol')&paired.condition.isin(PANEL)].groupby(['track','condition']):report.append(f'- {t}, {N[m]}: {int((g.delta_rmse_mean < -1e-10).sum())}/{len(g)}. Seed 평균 부호이며 통계적 유의성 검정이 아니다.')
 report+=['## 검증 및 해석',f'동일 query/실제값/식별자 검사 {len(checks)}건 통과. 기존 예측파일 {len(reused)}개를 원본/hash와 대조했다. Uni-Mol feature provider로 원래 Known seed1을 재학습하여 원본 예측과 완전히 일치했다.', '표의6개 primary 표현은 모두 관계 손실OFF, 동일512→32 prior와 observer/dynamics/분할/validation 선택/120epoch 예산을 사용한다. 기존 full MHAP-PK는 두 관계 손실ON인 문맥 비교이며 표현만 바꾼 통제조건이 아니다.', 'ChemBERTa MLM/MTR는 frozen384차원 평균 token pooling이다. Train-fold drug 평균/표준편차로 정규화 후128개0으로 채운다. 정보 손실은 없지만 첫 층의4096개 가중치는 비활성이므로 실효 입력 용량까지 같다고 주장하지 않는다. Uni-Mol/ECFP512차원과 allocated network 수는 같다.', '단일 약물 학습에서는 분자표현이 train normalization 후0이 된다. Target-only 기준은 동일 원래 조건이며 각 encoder를 독립적으로 학습한 비교로 해석하지 않는다.', 'Drug-ID는 약물당512개 학습 embedding 가중치가 추가된다. No-molecule의 입력 projection weights는 입력0이라 활성화되지 않는다. 따라서 모든 입력 대조군의 실효 파라미터 수까지 같다고 주장하지 않는다. Drug-ID의 미관측 약물은 train-defined fallback이며 test-drug 농도/ID를 prior 학습에 사용하지 않는다. No-molecule은 chemistry-derived relation까지OFF다.', 'Joint는 PK 학습자료에서 drug/source가 함께 빠진 조건이다. Pretrained encoder가 해당 분자 구조를 본 적 없다는 보장은 없으며, 독립 외부 검증이 아니다. Uni-Mol 원 checkpoint byte hash는 기존 cache기록에 없어 미확인 상태로 유지한다.', 'Representation별 rank는 평가자료의 기술적 결과다. Test 순위로 원고의 최종 모델을 교체하지 않는다. Relation-on ChemBERTa 확장은 이번 relation-off 비교와 별도 실험으로 등록해야 한다.', 'Train은 기존 k=0,1,2,4 분포를 유지하고 k=3은 inference-only다. 모든k는 같은 t4이후 query를 사용한다. SD는 학습seed 변동이며 임상 예측구간이 아니다.','## 선택 epoch와 비용',select.to_markdown(index=False)]
 report+=interpretation(cs,ds,paired)
 (O/'REPORT_KO.md').write_text('\n\n'.join(report))
 manifest=dict(status='completed',metric_version='participant-MSE then source/drug hierarchical RMSE; canonical mhap_pk.metrics',metric_source_sha256=sha(R/'src/mhap_pk/metrics.py'),new_fits=530,inference_only=300,parity_regression_fit=1,seeds=list(range(1,11)),query_checks=len(checks),source_snapshot_sha256=sha(B/'source_snapshot.tar.gz'),previous_predictions_sha256=sha(previous),low_data_predictions_sha256=sha(previous2),files={x.name:sha(x) for x in O.iterdir() if x.is_file() and x.name!='analysis_manifest.json'})
 (O/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2));print(cs[cs.k==2][['track','condition','sb_hrmse_mean','sb_hrmse_std']].to_string(index=False))
if __name__=='__main__':main()
