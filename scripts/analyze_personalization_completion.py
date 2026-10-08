"""Prediction-derived metrics and qualitative review for the completed registered study."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.quality_metrics import extended_metrics
from mhap_pk.metrics import metric
from mhap_pk.data.traceable import load_profiles
from mhap_pk.training.run_io import digest,save_json
root=Path(sys.argv[1]).resolve();assert json.loads((root/'status.json').read_text())['status']=='completed'
out=R/'results/summaries'/('personalization_completion_'+root.name);out.mkdir(exist_ok=False);figdir=out/'figures';figdir.mkdir()
lock=json.loads((root/'method_locked.json').read_text());selected=lock['selected_condition'];plan=json.loads((root/'plan.json').read_text())
frames=[];diagnostics=[];hist=[];selections=[]
for d in sorted(root.glob('*__s*')):
 if not d.is_dir():continue
 manifest=json.loads((d/'manifest.json').read_text());assert manifest['status']=='completed';assert digest(d/'predictions.parquet')==manifest['prediction_hash']
 frames.append(pd.read_parquet(d/'predictions.parquet'));diagnostics.append(pd.read_parquet(d/'diagnostics.parquet'))
 h=pd.read_csv(d/'history.csv');h['partition']=manifest['partition'];h['seed']=manifest['seed'];hist.append(h)
 s=pd.read_csv(d/'selection.csv');s['partition']=manifest['partition'];s['seed']=manifest['seed'];selections.append(s)
p=pd.concat(frames,ignore_index=True);dg=pd.concat(diagnostics,ignore_index=True);hs=pd.concat(hist,ignore_index=True);ss=pd.concat(selections,ignore_index=True)
p=p.rename(columns={'observed_log':'y_true_log','predicted_log':'y_pred_log'});keys=['partition','role','protocol','k','seed','source_id','drug_id','participant_id','profile_id','query_id'];assert not p.duplicated(keys+['condition']).any();assert np.isfinite(p[['y_true_log','y_pred_log']]).all().all();assert np.isfinite(np.exp(p[['y_true_log','y_pred_log']])).all().all();assert (p.query_time>p.cutoff.fillna(-np.inf)).all()
for _,g in p.groupby(['partition','role','protocol','k','seed']):
 expected=None
 for c,x in g.groupby('condition'):
  q=x.set_index(keys)[['query_time','y_true_log']].sort_index()
  if expected is None:expected=q
  else:pd.testing.assert_frame_equal(q,expected)
diagnostic_keys=['partition','role','protocol','k','seed','source_id','drug_id','participant_id','profile_id','condition'];p=p.merge(dg[diagnostic_keys+['fallback_flag']],on=diagnostic_keys,validate='many_to_one');p['dataset_id']='source_traceable_20260925';p['split_id']='source_traceable_20260925';p['run_id']=root.name+'__'+p.partition+'__s'+p.seed.astype(str);p['metric_version']='canonical_source_drug_participant_log_rmse';p.to_parquet(out/'predictions.parquet',index=False);dg.to_parquet(out/'diagnostics.parquet',index=False);hs.to_csv(out/'training_curves.csv',index=False);ss.to_csv(out/'checkpoint_selections.csv',index=False)
# Replay the old 120-epoch budget to separate algorithm changes from bookkeeping.
legacy_checks=[]
for seed in range(1,6):
 old_path=R/f'results/runs/PERSONALIZATION/shared_20260927/known__s{seed}/predictions.parquet'
 if old_path.exists():
  old=pd.read_parquet(old_path);old=old[(old.condition=='prior')&(old.protocol=='first_k')&(old.k==2)].rename(columns={'observation_id':'query_id','y_pred_lnc':'old_prediction'})
  now=p[(p.partition=='original')&(p.role=='assessment')&(p.condition=='prior_budget120')&(p.protocol=='first_k')&(p.k==2)&(p.seed==seed)]
  link=['seed','source_id','drug_id','participant_id','profile_id','query_id'];matched=now.merge(old[link+['old_prediction']],on=link,validate='one_to_one');assert len(matched)==len(now)==len(old)
  difference=float((matched.y_pred_log-matched.old_prediction).abs().max());legacy_checks.append(dict(seed=seed,queries=len(matched),max_absolute_difference=difference))
save_json(out/'legacy_prior_replay.json',dict(checks=legacy_checks,interpretation='Same first120-epoch trajectory expected; differences reported, never silently forced.'))
metrics=[]
for key,g in p.groupby(['partition','role','protocol','k','condition','seed']):metrics.append(dict(zip(['partition','role','protocol','k','condition','seed'],key))|extended_metrics(g))
m=pd.DataFrame(metrics);m.to_csv(out/'metrics_22_by_seed.csv',index=False)
# Recomputed selection primary score must equal the score that picked checkpoints.
check=m[(m.role=='selection')&(m.protocol=='first_k')&(m.k==2)].merge(ss,on=['partition','seed','condition'],validate='one_to_one');assert np.allclose(check.sb_hrmse,check.selection_sb_hrmse,atol=1e-10)
primary=m[(m.partition=='original')&(m.role=='assessment')&(m.protocol=='first_k')&(m.k==2)]
cols=['sb_hrmse','log_rmse','log_mae','within_twofold_pct','signed_bias','p90_abs_log_error']
main=primary.groupby('condition')[cols].agg(['mean','std']);main.columns=['_'.join(c) for c in main.columns];main['selected_before_original_assessment']=main.index==selected;main.to_csv(out/'primary_table.csv')
inner=m[(m.partition!='original')&(m.role=='assessment')];inner.groupby(['partition','condition'])[cols].agg(['mean','std']).to_csv(out/'inner_fold_table.csv')
# Per-source/drug and paired prior effects, maintaining drug-aware identities.
x=p[(p.partition=='original')&(p.role=='assessment')&(p.protocol=='first_k')&(p.k==2)].copy();x['sq']=(x.y_pred_log-x.y_true_log)**2;drug,source=metric(x);drug.to_csv(out/'drug_by_seed.csv',index=False);source.to_csv(out/'source_by_seed.csv',index=False)
src=source.groupby(['condition','source_id']).sb_hrmse.mean().unstack(0);effects=[];rng=np.random.default_rng(270927);sample=rng.integers(0,len(src),(10000,len(src)))
for name in src.columns:
 difference=(src[name]-src.prior).to_numpy();boot=difference[sample].mean(1);effects.append(dict(condition=name,delta_sb_hrmse=difference.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),sources_improved=int((difference< -1e-10).sum()),sources=len(src)))
pd.DataFrame(effects).to_csv(out/'paired_source_effects.csv',index=False)
leave=[]
for omitted in src.index:
 for c in src.columns:
  keep=src.drop(index=omitted);leave.append(dict(omitted_source=omitted,condition=c,delta_sb_hrmse=float((keep[c]-keep.prior).mean())))
pd.DataFrame(leave).to_csv(out/'leave_one_source.csv',index=False)
identity=['source_id','drug_id','participant_id','profile_id'];prof=x.groupby(['condition','seed']+identity).sq.mean().reset_index();prior=prof[prof.condition=='prior'].drop(columns='condition').rename(columns={'sq':'prior_mse'});prof=prof.merge(prior,on=['seed']+identity,validate='many_to_one');prof['delta_mse']=prof.sq-prof.prior_mse;prof.to_csv(out/'profile_adaptation_by_seed.csv',index=False)
pavg=prof.groupby(['condition']+identity)[['sq','prior_mse','delta_mse']].mean().reset_index();pavg.to_csv(out/'profile_adaptation_seed_mean.csv',index=False)
negative=pavg.groupby('condition').agg(profiles=('delta_mse','size'),worsened_pct=('delta_mse',lambda v:100*np.mean(v>1e-10)),mean_delta_mse=('delta_mse','mean'),median_delta_mse=('delta_mse','median'),p90_delta_mse=('delta_mse',lambda v:v.quantile(.9)));negative.to_csv(out/'negative_adaptation.csv')
pavg.groupby(['condition','drug_id']).agg(profiles=('delta_mse','size'),mean_delta_mse=('delta_mse','mean'),worsened_pct=('delta_mse',lambda v:100*np.mean(v>1e-10))).to_csv(out/'drug_adaptation.csv')
# Descriptive selection-to-assessment gaps are not used to choose the winner.
gaps=m[(m.partition=='original')&(m.protocol=='first_k')&(m.k==2)].pivot(index=['seed','condition'],columns='role',values='sb_hrmse');gaps['assessment_minus_selection']=gaps.assessment-gaps.selection;gaps.to_csv(out/'selection_assessment_gap.csv')
lat=dg.groupby(['partition','role','protocol','k','condition','seed']).agg(batch_seconds=('batch_seconds','first'),profiles=('profiles_in_batch','first'),failure_count=('fallback_flag','sum'));lat['ms_per_profile']=1000*lat.batch_seconds/lat.profiles;lat.to_csv(out/'inference_cost.csv')
ss['at_budget_boundary']=ss.epoch==np.where(ss.condition=='prior',720,360);ss.to_csv(out/'checkpoint_selections.csv',index=False)
# PDF, standalone figures, and cases selected by predefined median/extreme rules.
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
pdf=PdfPages(out/'analysis_report.pdf')
def savefig(fig,name):
 fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');fig.savefig(figdir/(name+'.png'),dpi=150,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title('MHAP-PK: completed personalization diagnostics\nInternal Known-drug evaluation; five seeds',fontsize=17,pad=20)
show=main[['sb_hrmse_mean','sb_hrmse_std','log_rmse_mean','log_mae_mean','within_twofold_pct_mean']].round(4)
tab=ax.table(cellText=show.values,rowLabels=show.index,colLabels=['SB-hRMSE','Seed SD','Log RMSE','Log MAE','Twofold %'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.8)
ax.text(0,.08,f'Locked condition from inner assessments: {selected}\n80 neural fits; 3 inner folds x 5 seeds + original split x 5 seeds.\nOriginal test was previously used in development. This is not external validation.\nMAP controls are fixed-noise penalized fits, not population NLME.',transform=ax.transAxes);savefig(fig,'main_table')
fig,axes=plt.subplots(2,2,figsize=(11.7,8.3))
for ax,condition in zip(axes.flat,['prior','parameter','state','combined']):
 for seed,g in hs[(hs.partition=='original')&(hs.condition==condition)].groupby('seed'):ax.plot(g.epoch,g.selection_sb_hrmse,label=f'seed {seed}',alpha=.8)
 ax.set(title=condition,xlabel='Epoch',ylabel='Selection SB-hRMSE');ax.legend(fontsize=7)
fig.suptitle('Learning-budget audit (selection split only)');fig.tight_layout();savefig(fig,'learning_budget')
fig,axes=plt.subplots(1,2,figsize=(11.7,5))
for partition,g in inner.groupby('partition'):
 v=g.groupby('condition').sb_hrmse.mean();axes[0].plot(v.index,v.values,marker='o',label=partition)
axes[0].tick_params(axis='x',rotation=70);axes[0].set(ylabel='Inner assessment SB-hRMSE',title='Split consistency');axes[0].legend()
for name in ['combined',selected]:
 values=pavg[pavg.condition==name].delta_mse
 axes[1].plot(np.sort(values),np.arange(1,len(values)+1)/len(values),label=name)
axes[1].axvline(0,color='black',ls='--');axes[1].set(xlabel='Profile MSE change vs same prior',ylabel='Empirical cumulative fraction',title='Original assessment: negative adaptation');axes[1].legend();fig.tight_layout();savefig(fig,'consistency_and_harm')
fig,ax=plt.subplots(figsize=(11.7,6));dd=drug.groupby(['drug_id','condition']).sb_hrmse.mean().unstack();display=list(dict.fromkeys(['prior','combined',selected,'map_cl_vc','fixed_state']));dd[display].plot.bar(ax=ax);ax.set(ylabel='Drug-balanced participant log RMSE',title='Drug-specific original assessment (5-seed score means)');fig.tight_layout();savefig(fig,'drug_comparison')
fig,axes=plt.subplots(1,2,figsize=(11.7,5))
for protocol in ['first_k','common_origin']:
 h=m[(m.partition=='original')&(m.role=='assessment')&(m.protocol==protocol)]
 ax=axes[0 if protocol=='first_k' else 1]
 for c in display:
  v=h[h.condition==c].groupby('k').sb_hrmse.mean();ax.plot(v.index,v.values,marker='o',label=c)
 ax.set(title=protocol,xlabel='Support blocks',ylabel='SB-hRMSE');ax.set_xticks([0,1,2,4]);ax.legend(fontsize=7)
fig.tight_layout();savefig(fig,'history_design')
ps,_,_=load_profiles();lookup={tuple(str(p[c]) for c in identity):p for p in ps};assert len(lookup)==len(ps)
candidate=selected if selected!='prior' else 'combined';pv=pavg[pavg.condition==candidate].copy();cases=[]
for label,row in [('median_error',pv.sort_values('sq').iloc[len(pv)//2]),('lowest_delta',pv.sort_values('delta_mse').iloc[0]),('highest_delta',pv.sort_values('delta_mse').iloc[-1])]:cases.append(dict(rule=label,**{c:row[c] for c in identity},delta_mse=row.delta_mse))
pd.DataFrame(cases).to_csv(out/'case_selection.csv',index=False)
fig,axes=plt.subplots(3,1,figsize=(10,11))
for ax,case in zip(axes,cases):
 key=tuple(str(case[c]) for c in identity);profile=lookup[key];g=x[x.seed==1]
 for col in identity:g=g[g[col]==case[col]]
 assert g.drug_id.nunique()==1
 target=g[g.condition=='prior'].sort_values('query_time');ax.scatter(target.query_time,np.exp(target.y_true_log),color='black',marker='o',s=35,label='Future measured',zorder=5)
 support=profile['episodes'][2].support;ax.scatter(support.time,np.exp(support.target_log_dv),marker='s',color='orange',edgecolors='black',s=55,label='Support',zorder=6)
 for name in list(dict.fromkeys(['prior',candidate,'map_cl_vc'])):
  line=g[g.condition==name].sort_values('query_time');ax.plot(line.query_time,np.exp(line.y_pred_log),marker='.',label=name)
 for i,event in enumerate(profile['events']):ax.axvline(event[0],color='gray',alpha=.4,ls=':',label='Dose event' if i==0 else None)
 ax.axvline(support.time.max(),color='orange',ls='--',alpha=.5);ax.set(yscale='log',xlabel='Time (h)',ylabel='Concentration (umol/L)',title=f"{case['rule']}: {case['drug_id']} | {case['participant_id']} | queries={len(target)}")
 ax.legend(fontsize=8,ncol=3)
fig.suptitle('Seed 1 curves; selection by five-seed profile MSE\nLines connect query predictions, not dense solver trajectories');fig.tight_layout(rect=(0,0,1,.95));savefig(fig,'cases')
# Per-drug coverage atlas: median and extreme changes, deduplicated within drug.
atlas=[]
for drug_name,group in pv.groupby('drug_id'):
 for rule,row in [('median_error',group.sort_values('sq').iloc[len(group)//2]),('lowest_delta',group.sort_values('delta_mse').iloc[0]),('highest_delta',group.sort_values('delta_mse').iloc[-1])]:
  key=tuple(str(row[c]) for c in identity)
  if any(tuple(str(a[c]) for c in identity)==key for a in atlas):continue
  atlas.append(dict(rule=rule,**{c:row[c] for c in identity},delta_mse=float(row.delta_mse)))
pd.DataFrame(atlas).to_csv(out/'drug_case_atlas_selection.csv',index=False)
for page in range(0,len(atlas),4):
 fig,axes=plt.subplots(2,2,figsize=(11.7,8.3))
 for ax,case in zip(axes.flat,atlas[page:page+4]):
  key=tuple(str(case[c]) for c in identity);profile=lookup[key];g=x[x.seed==1]
  for col in identity:g=g[g[col]==case[col]]
  assert g.drug_id.nunique()==1
  target=g[g.condition=='prior'].sort_values('query_time');support=profile['episodes'][2].support
  ax.scatter(target.query_time,np.exp(target.y_true_log),color='black',s=25,label='Future measured',zorder=5)
  ax.scatter(support.time,np.exp(support.target_log_dv),marker='s',color='orange',edgecolors='black',s=40,label='Support',zorder=6)
  for name in list(dict.fromkeys(['prior',candidate,'map_cl_vc'])):
   line=g[g.condition==name].sort_values('query_time');ax.plot(line.query_time,np.exp(line.y_pred_log),marker='.',label=name)
  for event in profile['events']:ax.axvline(event[0],color='gray',alpha=.4,ls=':')
  ax.axvline(support.time.max(),color='orange',ls='--',alpha=.5);ax.set(yscale='log',xlabel='Time (h)',ylabel='Concentration (umol/L)',title=f"{case['drug_id']} | {case['rule']}\nqueries={len(target)}, delta MSE={case['delta_mse']:.3g}")
  ax.legend(fontsize=6)
 for ax in list(axes.flat)[len(atlas[page:page+4]):]:ax.axis('off')
 fig.suptitle('Per-drug case atlas: seed 1, query-point connections, gray dosing events');fig.tight_layout(rect=(0,0,1,.96));savefig(fig,f'drug_cases_{page//4+1:02d}')
# Record sampling design so sparse profiles are visible rather than excluded.
sampling=[]
for role,idxs in plan['partitions']['original'].items():
 for idx in idxs:
  pp=ps[idx];support=pp['episodes'][2].support;cutoff=float(support.time.max())
  sampling.append({c:pp[c] for c in identity}|dict(role=role,queries=len(pp['query']),support_span=float(support.time.max()-support.time.min()),median_forecast_horizon=float((pp['query'].time-cutoff).median())))
sp=pd.DataFrame(sampling);sp.to_csv(out/'sampling_design.csv',index=False)
sp.groupby(['role','drug_id']).agg(profiles=('profile_id','size'),participants=('participant_id','nunique'),queries=('queries','sum'),median_horizon=('median_forecast_horizon','median')).to_csv(out/'drug_sample_counts.csv')
pdf.close()
# Original whole-method reference comparison preserved as context, not mixed fits.
reference=R/'results/summaries/comparison_completion_20260927/human_main_k2.csv'
if reference.exists():pd.read_csv(reference).to_csv(out/'previous_whole_method_reference.csv',index=False)
innermeans=inner.groupby('condition').sb_hrmse.mean().sort_values();priorends=ss[(ss.condition=='prior')&(ss.epoch==720)]
report=f'''# 개인화 방법론 보완: 실험·분석 완료

## 완료한 범위
- 3개 training-only 내부 분할 × 5 seeds, 원래 Known 분할 × 5 seeds: 총20개 fit job, prior20회 + adapter60회 = **80회 신경망 학습**.
- 공통 prior, parameter-only, state-only, combined를 독립 학습. Prior는 최대720epoch, adapter는360epoch.
- 단순 고정 상태 보정과 amplitude / CL·Vc / full-active MAP을 같은 prior에서 비교. 고정 gain5후보, MAP scale각4후보를 selection split으로 선택했다.
- Original 분할은 first-k와 common-origin, k0/1/2/4를 모두 평가했다. 조건마다 22개 지표, 약물·source·profile별 악화, 비용과 실패율을 저장했다. Latency는 여러 학습 worker가 공유한 CPU에서 측정한 진단값이며 전용 배포장치 benchmark가 아니다.
- 이전 보수적 업데이트25회 실험도 완료되어 보존되어 있다. 이번80회와 합쳐105회 새 학습이지만 서로 다른 실험으로 구분한다.

## 사전 선택 및 내부 평가
내부 assessment의 평균 주 지표로 고정한 조건: **{selected}**. 기존 test를 이미 개발에서 사용한 이력이 있으므로 원래 분할의 assessment는 독립 검증이 아니다.

### 내부 분할 평균 SB-hRMSE
{innermeans.to_frame('mean').to_markdown(floatfmt='.6f')}

### 원래 Known 분할 재평가 (k2)
{main[['sb_hrmse_mean','sb_hrmse_std','log_rmse_mean','log_mae_mean','within_twofold_pct_mean']].to_markdown(floatfmt='.6f')}

### Prior 대비 profile 악화
{negative.to_markdown(floatfmt='.4f')}

## 해석과 결정

**이번 실험은 공동 보정의 최종 우월성을 지지하지 않았다.** 내부 model selection에서는 combined가 선택됐지만, 원래 Known assessment의 SB-hRMSE는 prior {main.loc['prior','sb_hrmse_mean']:.6f}, combined {main.loc['combined','sb_hrmse_mean']:.6f}였다. Assessment 결과를 본 뒤 parameter 또는 prior로 모델 선택 결과를 바꾸지 않는다. 대신 combined를 개선된 최종 개인화 방법으로 채택할 근거가 부족하다는 결론을 남긴다.

Prior 학습 예산120→720의 선택 범위 확대에서는 {main.loc['prior_budget120','sb_hrmse_mean']:.6f}→{main.loc['prior','sb_hrmse_mean']:.6f}로 개선됐다. 이것은 새 적응 구조의 기여가 아니다. 기존120epoch의 query 예측은 legacy_prior_replay.json에서 별도로 대조했다.

Combined의5seed 평균 profile MSE는 {negative.loc['combined','worsened_pct']:.2f}%의 profile에서 악화했다. Participant 수와 profile 수는 다르며 이는101개 약물별 profile의 기술통계다. 가장 큰 source 차이와 모든 leave-one-source 민감도는 source_by_seed.csv 및 leave_one_source.csv에 기록했다. Source를 사후 제외하거나 primary metric을 바꾸지 않는다. 특히 fluorouracil은 test profile1개/query1개인 작은 source이므로 한 사례가 source-balanced 점수에 미치는 영향을 별도로 해석한다. 생물학적 원인을 이 사례만으로 확정하지 않는다.

이번 결과로 방법론 신규성·외부 일반화·임상적 유효성을 확정할 수 없다. 실험 완료와 방법론 논문의 근거 확보는 별개다. 분할에 따라 보정의 방향이 달라지고 일부 prior fit의 budget boundary가 남아 있다는 사실까지 포함해 이번 검증을 종료한다.
신규 구조를 억지로 최종 모델로 채택하지 않는다. Locked condition과 모든 사전 등록 대조군을 함께 보고한다. prior_budget120은 적응법이 아니라 학습 예산 대조군이므로 그 행의 prior 대비 차이는 개인화 효과로 해석하지 않는다. 원래 Sequential의 relation-on native 학습과 이번 frozen-prior relation-off 통제는 다른 조건이다. `previous_whole_method_reference.csv`의 외부 비교 구현 결과는 맥락 자료이며 이번 ablation 행과 동일 학습 조건인 것처럼 병합하지 않는다.

총20개 prior fit 중 마지막 epoch720이 선택된 것은 {len(priorends)}개다. 마지막 epoch가 선택되지 않았다고 전역 수렴을 증명하는 것은 아니다. `checkpoint_selections.csv` 및 `learning_budget` 그림에서 확인한다. Prior-budget120은 같은 optimization trajectory의 처음120epoch 내 best checkpoint이다.

MAP noise0.3/scale 후보는 규약으로 정한 penalized inference이며 train으로 추정한 population random effects가 아니다. 따라서 NLME 또는 selective prior flattening 원논문 재현으로 명명하지 않는다. Joint 평가와 독립 외부 자료 확보는 이번 Known-personalization 검증 범위에 포함하지 않았다. 기존 Joint 결과를 이 연구의 개선 결과로 바꾸지 않는다.

연구 기여가 확보되었다는 결론은 단지 실험이 끝났다는 이유로 내리지 않는다. 평균 개선, 분할 간 일관성, 강한 단순 기준 대비 추가 효과를 함께 판단해야 한다.

## 검증
- 참가자/source 연결 그룹을 split 사이에서 차단하고 모든 내부 split은 기존 training 참가자 안에서만 구성했다. 모든 training split에11약물이 존재한다.
- 모든 condition의 query ID·시각·실제값 일치, support 이후 query, 중복키/NaN/Inf/선형변환 overflow 검사를 통과했다.
- 저장한 query 예측으로 재집계한 selection SB-hRMSE가 checkpoint 선택 점수와 일치했다.
- 지표는 source/drug/participant의 기존 canonical 함수와22metric 함수를 사용했다. Seed 평균 score와 seed ensemble을 혼합하지 않았다.
- `paired_source_effects.csv`는5seed 평균 source 차이를10000회 source bootstrap한 탐색 구간이다. 독립 환자 수를 seed/profile 수로 부풀리지 않았으며 재사용/모델선택 편향을 제거하는 구간도 아니다.
- 곡선은 drug-aware key로 선택, 동일 seed1, support/미래측정/투약을 구분한다. 중앙 오차 사례·최소 MSE 변화·최대 MSE 변화 사례를 사전 규칙으로 선택했다. Delta가 음수이면 개선, 양수이면 악화다. 약물의 모든 사례가 개선된 경우 최대 delta 사례도 악화 사례라고 단정하지 않는다.

## 산출물
`analysis_report.pdf`, `primary_table.csv`, `metrics_22_by_seed.csv`, `inner_fold_table.csv`, `drug_by_seed.csv`, `paired_source_effects.csv`, `negative_adaptation.csv`, `selection_assessment_gap.csv`, `inference_cost.csv`, 원 query 예측·진단 및 figure 원본.

원고의 기존 수치는 보존한다. 이 분석을 반영하지 않은 기존 PDF를 이번 실험까지 포함한 원고라고 표현하지 않는다.
'''
(out/'REPORT_KO.md').write_text(report)
save_json(out/'analysis_manifest.json',dict(status='completed',run_directory=str(root.relative_to(R)),locked_condition=selected,source_plan_sha256=digest(root/'plan.json'),analysis_code_sha256=digest(Path(__file__)),prediction_rows=len(p),metric_rows=len(m),nan_or_missing_query=False,selection_score_recomputed=True,files={str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file()}))
print(out);print(main.to_string())
