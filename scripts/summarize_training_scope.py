"""Reorganize existing evidence without relabelling pooled fits as single-drug fits."""
from pathlib import Path
import json,hashlib,sys
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from mhap_pk.evaluation.quality_metrics import extended_metrics
S=R/'results/summaries/ten_seeds_20261001';O=R/'results/summaries/training_scope_20261001';O.mkdir(exist_ok=True,parents=True)
p=pd.read_parquet(S/'canonical_predictions.parquet',filters=[('track','==','Known-drug'),('k','==',2)])
p=p[p.condition.isin(['single_drug','poppk_vi_map'])]
checks=[]
for seed,g in p.groupby('seed'):
 a=g[g.condition=='single_drug'].set_index('observation_id').sort_index();b=g[g.condition=='poppk_vi_map'].set_index('observation_id').sort_index()
 assert a.index.is_unique and a.index.equals(b.index)
 for c in ['source_id','drug_id','participant_id','profile_id','query_time','y_true_log']:assert np.array_equal(a[c],b[c])
 checks.append(dict(seed=int(seed),queries=len(a),same_queries_and_targets=True))
assert set(p.seed)==set(range(1,11))
metrics=pd.DataFrame([dict(condition=c,seed=int(seed),**extended_metrics(g)) for (c,seed),g in p.groupby(['condition','seed'])]);metrics.to_csv(O/'single_drug_comparison_by_seed.csv',index=False)
d=pd.read_csv(S/'drug_metrics_summary.csv');d=d[(d.track=='Known-drug')&(d.k==2)&d.condition.isin(['single_drug','poppk_vi_map'])]
counts=pd.read_csv(R/'results/summaries/single_drug_20261001/partition_counts.csv');d=d.merge(counts[['drug_id','train_participants','validation_participants']],on='drug_id',validate='many_to_one');d.to_csv(O/'single_drug_comparison.csv',index=False)
names={'single_drug':'MHAP-PK — single-drug','poppk_vi_map':'PopPK (MAP) — drug-specific'}
rows=['# 학습 범위별 비교','', '현재 확보된 약물별 독립 학습 결과만 비교한다. Seeds 1–10, Known-drug, k=2. 기존 DCM/DeepCt/Uni-PK 점수는 pooled training이므로 이 표에 개별 학습 결과로 넣지 않는다.','', '| Drug | Training participants | Validation participants | MHAP-PK single-drug log-RMSE | PopPK (MAP) log-RMSE |','|---|---:|---:|---:|---:|']
wins=0
for drug,g in d.groupby('drug_id'):
 g=g.set_index('condition');best=g.rmse.min();wins+=int(g.loc['single_drug','rmse']<g.loc['poppk_vi_map','rmse']);v=[]
 for c in names:
  r=g.loc[c];x=f'{r.rmse:.4f} ± {r.rmse_sd:.4f}';v.append('**'+x+'**' if r.rmse==best else x)
 rows.append(f'| {drug} | {int(g.iloc[0].train_participants)} | {int(g.iloc[0].validation_participants)} | '+' | '.join(v)+' |')
rows+=['',f'MHAP-PK single-drug의 평균 drug log-RMSE가 낮은 약물: {wins}/11. 이는 point-estimate 비교이며 유의성 주장이 아니다. Drug-level MAE/within-twofold/bias는 CSV 및 Excel에 포함한다.','', '이 표는 저자료량 전이의 증거가 아니다. 학습 참가자 수를 제한하고 target-only와 donor-augmented fitting을 비교하는 새 통제실험이 필요하다. 현재 결과는 전 약물의 기존 학습 참가자를 사용했다.']
(O/'REPORT_KO.md').write_text('\n'.join(rows))
with pd.ExcelWriter(O/'single_drug_comparison.xlsx') as w:
 d.to_excel(w,sheet_name='Drug metrics',index=False);metrics.to_excel(w,sheet_name='Cohort metrics by seed',index=False);counts.to_excel(w,sheet_name='Participant counts',index=False)
(O/'manifest.json').write_text(json.dumps(dict(status='completed_reaggregation_only',training_runs_added=0,checks=checks,primary_k=2,seeds=list(range(1,11)),input_sha256=hashlib.sha256((S/'canonical_predictions.parquet').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),not_completed=['Single-drug DCM','Single-drug DeepCt','Single-drug Uni-PK','Matched low-training-data transfer on current cohort']),indent=2))
print('Matched query checks passed; single-drug MHAP lower mean RMSE:',wins,'/11')
