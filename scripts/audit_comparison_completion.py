"""Verify existing human comparisons; keep native public examples in a separate panel."""
from pathlib import Path
import hashlib
import json
import sys
import time
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from recompute_evaluation_metrics import recompute
OUT=ROOT/'results/summaries/comparison_completion_20260927'
NATIVE=ROOT/'results/runs/OFFICIAL_REPRODUCTION/20260927/full_public_examples'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def human_audit():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'results/summaries/source_traceable_20260925/canonical_predictions.parquet'
    x=pd.read_parquet(path)
    assert set(x.dataset_id)=={'source_traceable_20260925'}
    assert set(x.split_id)=={'source_traceable_20260925'}
    scores=recompute(x.rename(columns={'y_true_log':'y_true_lnc','y_pred_log':'y_pred_lnc'}))
    previous=pd.read_csv(ROOT/'results/summaries/source_traceable_20260925/seed_metrics.csv')
    keys=['track','protocol','k','condition','seed']
    joined=scores.merge(previous,on=keys,validate='one_to_one',suffixes=('_new','_old'))
    assert len(joined)==len(scores)==len(previous)
    metric_names=['sb_hrmse','log_rmse','log_mae','signed_bias','within_twofold_pct','p90_abs_log_error']
    differences={m:float(abs(joined[m+'_new']-joined[m+'_old']).max()) for m in metric_names}
    assert max(differences.values())<1e-10
    scores.to_csv(OUT/'verified_seed_metrics.csv',index=False)
    main=['molecular_relation','dcm','deepct','unipk','poppk_vi_map']
    identity=['fold','source_id','drug_id','participant_id','profile_id','observation_id','query_time']
    reference={}
    for vals,g in x[x.condition=='molecular_relation'].groupby(['track','protocol','k','seed']):
        reference[vals]=g[identity+['y_true_log']].sort_values(identity).reset_index(drop=True)
    checks=[]
    for vals,g in x[x.condition.isin(main)].groupby(keys):
        track,protocol,k,condition,seed=vals
        expected=reference[(track,protocol,k,seed)]
        actual=g[identity+['y_true_log']].sort_values(identity).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
        checks.append(dict(zip(keys,vals))|dict(queries=len(g),identical_query_and_truth=True))
    pd.DataFrame(checks).to_csv(OUT/'query_coverage.csv',index=False)
    full=scores[scores.condition.isin(main)]
    table=full.groupby(keys[:-1])[metric_names].agg(['mean','std']).reset_index()
    table.columns=['_'.join(c).rstrip('_') for c in table.columns]
    table.to_csv(OUT/'human_main_all_protocols.csv',index=False)
    table[(table.protocol=='first_k')&(table.k==2)].to_csv(OUT/'human_main_k2.csv',index=False)
    delivery_path=ROOT/'results/summaries/source_traceable_quality_20260926/delivery_manifest.json'
    delivery=json.loads(delivery_path.read_text())
    assert delivery['prediction_hash']==sha(path)
    checked=[]
    for relative,digest in delivery['artifacts'].items():
        artifact=ROOT/relative
        assert sha(artifact)==digest,relative
        checked.append(relative)
    registry=[
      ['MHAP-PK','completed','Known / Joint, seeds 1–5, k 0/1/2/4','Our existing human model; no new model training claimed'],
      ['DeepCt','completed_human_implementation','Known / Joint, seeds 1–5','Human implementation differs from native data/representation; native dummy separate'],
      ['Uni-PK','completed_human_implementation','Known / Joint, seeds 1–5','Human implementation separate from official rat checkpoint evaluation'],
      ['Deterministic DCM','completed','Known / Joint, seeds 1–5','Internal mechanistic control; not ME-DCM'],
      ['PopPK VI + MAP','completed','Known, seeds 1–5','Joint drug-specific population unavailable; never fit test population'],
      ['Lu Neural-ODE-PK','native_public_training_in_progress','Public simulation only','Genuine human matched-task integration remains unimplemented'],
      ['ME-DCM','missing_native_assets_and_human_implementation','none','Exact native clinical files/dependency files missing; human implementation unimplemented'],
      ['Selective prior flattening','unimplemented','none','No score; implementing literature mechanism remains separate work'],
      ['AICMET','unimplemented_context_protocol','none','No matched population-context evaluation registered'],
    ]
    pd.DataFrame(registry,columns=['method','status','scope','limitation']).to_csv(OUT/'completion_registry.csv',index=False)
    manifest=dict(status='passed',canonical_rows=len(x),seed_metric_cells=len(scores),
        paired_main_cells=len(checks),metric_differences=differences,prediction_sha256=sha(path),
        prior_quality_artifacts_verified=len(checked),extended_metric_count=delivery['metric_count'],
        script_sha256=sha(Path(__file__)),all_requested_literature_models_complete=False)
    (OUT/'human_verification.json').write_text(json.dumps(manifest,indent=2))


def report():
    OUT.mkdir(parents=True,exist_ok=True)
    native_rows=[]
    for path in sorted(NATIVE.glob('*/run_manifest.json')):
        manifest=json.loads(path.read_text())
        row=dict(run=path.parent.name,status=manifest['status'],elapsed_seconds=manifest.get('elapsed_seconds'))
        metrics=path.parent/'metrics.json'
        if metrics.exists():row.update(json.loads(metrics.read_text()))
        native_rows.append(row)
    pd.DataFrame(native_rows).to_csv(OUT/'native_public_status.csv',index=False)
    lu=[r for r in native_rows if r['run'].startswith('lu_') and r['status']=='completed']
    if lu:
        pd.DataFrame(lu).to_csv(OUT/'lu_public_fit_metrics.csv',index=False)
    registry_path=OUT/'completion_registry.csv'
    if registry_path.exists():
        registry=pd.read_csv(registry_path)
        if len(lu)==25:
            registry.loc[registry.method=='Lu Neural-ODE-PK','status']='native_public_completed_human_unimplemented'
        registry.to_csv(registry_path,index=False)
    table=pd.read_csv(OUT/'human_main_k2.csv')
    names={'molecular_relation':'MHAP-PK Sequential','dcm':'Deterministic DCM','deepct':'DeepCt (human task)','unipk':'Uni-PK (human task)','poppk_vi_map':'PopPK VI + MAP'}
    display=table[['track','condition','sb_hrmse_mean','sb_hrmse_std','log_rmse_mean','within_twofold_pct_mean']].copy()
    display['condition']=display.condition.map(names)
    verified=json.loads((OUT/'human_verification.json').read_text())
    text=f'''# 비교 실험 완료 범위 — 2026-09-27

## 사람 데이터의 기존 비교: 재검증 완료

출처 추적 가능한 11약물 cohort, 고정 split, seed 1–5를 사용했다. 이전 완료 run을 재검증한 것이며 새 학습 결과로 표시하지 않는다.
저장된 query 예측 {verified['canonical_rows']:,}행에서 {verified['seed_metric_cells']}개 seed 지표 조합을 다시 계산했다.
주 비교의 {verified['paired_main_cells']}개 조합에서 query ID·시각·실제값이 동일했다. 이전 22개 지표/정성 분석 산출물 {verified['prior_quality_artifacts_verified']}개 checksum도 확인했다.

{display.to_markdown(index=False,floatfmt='.5f')}

DeepCt/Uni-PK 행은 기존 human-task 구현 결과이다. 공식 원 데이터의 결과와 합산하거나 원 논문 완전 재현으로 표시하지 않는다.
Known에서 MHAP-PK가 SB-hRMSE는 낮지만 PopPK의 pooled log-RMSE 및 within-twofold가 더 좋다. 모든 지표에서 우월하다는 결론을 내리지 않는다.

## 이번 신규 공개 코드 실행

Lu: 5개 fold × 5개 model, 실제 29 epoch, 원 run.sh 설정. 현재 완료 {len(lu)}/25건.
Native split은 test 참가자의 첫 cycle을 training/validation에 제공한다. 우리 Known/Joint 평가와 동일하지 않다.
DeepCt: 원 notebook의 고정 final 구조 10개 member를 dummy 데이터에서 학습한다. 이전 native HPO NaN 실패는 보존하고 HPO를 건너뛴 점을 명시한다. 전체 원 논문 재현이 아니다.
Uni-PK: 9월 26일 완료된 277 rat curve/공식 5 checkpoint 평가 유지. 보고된 6개 지표 중 5개가 논문 반올림 값과 불일치하며 원인은 미확인이다.

{pd.DataFrame(native_rows)[['run','status']].to_markdown(index=False) if native_rows else '아직 실행 manifest 없음'}

## 남은 항목

전체 문헌 패널은 미완료다. 실제 ME-DCM, selective prior flattening, AICMET은 human matched-task 구현 결과가 없다. Lu도 native simulation 실행을 human 비교 완료로 간주하지 않는다.
ME-DCM/DeepCt/Lu의 비공개 원 연구 자료는 제공받지 못했다. 공개 예제 실행으로 그 자료의 결과를 대체하지 않는다.
기존 11약물 비교, 신규 native 실행, 미구현 항목을 completion_registry.csv에서 분리했다. 신규 실행 상태의 원본은 batch_manifest.json 및 개별 run_manifest.json이다.
'''
    (OUT/'REPORT_KO.md').write_text(text)


if __name__=='__main__':
    if '--watch' in sys.argv:
        for _ in range(1440):
            report()
            batch=json.loads((NATIVE/'batch_manifest.json').read_text())
            if batch['status']!='running':break
            try:
                import os
                os.kill(batch['pid'],0)
            except ProcessLookupError:
                (OUT/'watcher_failure.json').write_text(json.dumps({'status':'batch_process_missing','pid':batch['pid']}))
                break
            time.sleep(60)
    else:
        if '--report-only' not in sys.argv:human_audit()
        report()
