"""Describe verified results without changing the selected primary model."""
from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parents[1];S=R/'results/summaries/completion_suite_20261002'
manifest=json.loads((S/'completion_manifest.json').read_text());assert manifest['status']=='completed'
c=pd.read_csv(S/'cohort_summary.csv');c=c[c.k==2]
d=pd.read_csv(S/'drug_summary.csv');d=d[d.k==2]
names={'molecular_relation_120':'MoDE-PK (기존 GRU, 120 epochs)','molecular_relation_240':'MoDE-PK (240 epochs)','dcm_120':'DCM (120)','dcm_240':'DCM (240)','unipk_120':'Uni-PK (120)','unipk_240':'Uni-PK (240)','deepct_120':'DeepCt (120)','deepct_240':'DeepCt (240)','poppk_vi_map':'PopPK + MAP','last_mlp':'현재 관측 MLP','lstm':'LSTM encoder','deep_sets':'Deep Sets식 prefix pooling','attention':'Attention encoder','synthetic':'훈련 자료 기반 합성 증강','residual_centered':'실측값 중심 PK perturbation'}
x=c.set_index(['model','track'])
lines=['# MoDE-PK 추가 실험·검증 결과','',
'기존 대표 모델과 주요 평가표를 보존했다. 이번 작업은 학습 예산·관측 encoder·증강·분할·분자 표현·기전의 영향을 검증한다. 새 조건의 test 점수로 최종 모델을 다시 선택하지 않았다.','',
'## 완료 범위','',
'- 신규 1,020 fits: 240-epoch 확장 360, encoder 400, 증강 200 (원래 생성 방식 100 + residual-centered 대조군 100), 합성 기전 60 (초기 설계 30 + 독립 요인 설계 30). 기존 seed 1 예산 확장 40 fits를 재사용했다.',
'- 기존 CV/분할 4,340 fits와 분자 표현 실험은 다시 학습하지 않고 예측·hash·지표를 재검증했다.',
'- 기존 100개 MoDE-PK 체크포인트의 full 예측 재현 및 업데이트 경로별 추론을 완료했다. 이는 새로 학습한 ablation과 구분한다.',
'- 실제 cohort의 새로운 실험은 모두 Known-drug 및 9개 joint holdout folds, seeds 1–10, initial-observation k=0/1/2/3/4를 평가했다.',
'- 모든 약물별 결과는 `drug_summary.csv`, seed별 결과는 `drug_metrics_by_seed.csv`, 통합 Excel은 `Completion_results.xlsx`에 있다.','',
'## 결과의 구분','',
'| 질문 | 이번 처리 | 확인 위치 |',
'|---|---|---|',
'| 학습 시간을 늘리면 비교가 달라지는가? | 240-epoch 전체 패널, 10 seeds | cohort_summary.csv / optimization_audit.csv |',
'| 관측 encoder의 영향은? | MLP/LSTM/prefix pooling/attention 각각 재학습 | drug_summary.csv / controls_harm_by_drug_seed.csv |',
'| 합성 자료를 더하면 좋은가? | 원래 generator 및 실측 중심 perturbation, 각 100 fits | augmentation_teacher_fit.csv / centered_augmentation_verification.csv |',
'| 상태·파라미터 보정이 왜 실패하는가? | 알려진 생성 조건의 시뮬레이션 + 기존 100 checkpoint 경로 진단 | simulation_summary.csv / stage_drug_metrics.csv |',
'| 분할·분자 encoder의 영향은? | 기존 CV/비율·ChemBERTa 실험 재검증 | reuse_verification.json / molecular_metric_verification.csv |',
'| 본문 모델의 약물·예측 거리별 성능은? | 동일 pooled primary 예측 재집계 | primary_drug_metrics_summary.csv / Primary_drug_horizon_atlas.pdf |',
'',
'## k=2 요약','',
'아래는 SB-hRMSE의 10-seed 평균 ± SD다. 약물별 결과를 대신하지 않는다.','',
'| 모델/조건 | Known-drug | Joint drug–source |','|---|---:|---:|']
for model in names:
 vals=[]
 for track in ['Known-drug','Joint drug–source']:
  if (model,track) not in x.index:vals.append('적용하지 않음');continue
  z=x.loc[(model,track)];vals.append(f'{z.sb_hrmse_mean:.4f} ± {z.sb_hrmse_std:.4f}')
 lines.append('| '+names[model]+' | '+' | '.join(vals)+' |')
lines+=['','## 해석','',
'240-epoch 비교에서도 MoDE-PK는 Known 7/11, Joint 9/11 약물에서 가장 낮은 평균 log-RMSE였다. 그러나 Warfarin·Xanthohumol의 큰 holdout 오차는 해소되지 않았다. 약물별 최저 평균의 개수는 통계적 유의성 검정이 아니다.',
'LSTM은 기존 GRU보다 Known-drug의 11개 약물 모두에서 평균 log-RMSE가 낮았다. 그러나 Joint의 전체 SB-hRMSE는 거의 같고, 약물별 개선도 일관적이지 않았다. 이 결과는 관측 encoder 교체의 효과와 미관측 약물 전이의 한계를 구분한다.',
'합성 증강은 Known-drug에서 악화했다. 실제 training 자료에 맞춘 compartment teacher의 변동과 잡음을 추가하는 이번 방식이 일반적인 데이터 부족 해법이라고 볼 수 없다. 합성 자료는 새로운 독립 참가자나 분자가 아니며 모델의 구조적 오류를 반복할 수 있다. 생성용 teacher의 Known training 약물별 log-RMSE가 0.7228–2.5949이고 모든 잡음 SD가 상한 0.5에 도달했다. 따라서 이 결과는 낮은 생성 품질을 가진 해당 증강 방식의 한계이며 합성 데이터 일반의 효과를 부정하지 않는다.',
'같은 perturbation과 noise를 실측 training 농도에 중심화한 추가100회 실험은 Known SB-hRMSE 0.4614로 원래 GRU 0.4609에 가까워졌다. Joint는 2.3350으로 원래 2.2993보다 높았다. 생성기의 기준 농도 편향을 제거하자 Known의 큰 악화가 줄었지만, 증강 자체의 추가 이득은 확인되지 않았다.',
'Warfarin과 Xanthohumol은 동일 체크포인트의 분자 prior 단계에서도 큰 오차를 보였다. Warfarin에서는 state-only 경로가 full보다 낮은 오차였지만 Xanthohumol에서는 두 경로를 함께 사용하는 full이 더 좋았다. 시험 약물별로 유리한 경로를 사후 선택하는 방식은 검증된 개선 모델이 아니다.',
'ChemBERTa MLM은 Known에서 Uni-Mol보다 낮은 오차였지만 Joint에서는 높았다. 분자 표현의 장점을 두 평가 조건 모두에 일반화하지 않는다.',
'합성 기전 실험에서는 정확한 파라미터와 누락된 과거 투약, 낮은 잡음, 이른 관측 4개 조건에서 prior 0.3821, state-only 0.2364, combined 0.4740, CL/Vc MAP 0.4991의 latent log-RMSE를 얻었다. 반대로 늦은 관측에서는 MAP이 더 좋았다. 상태 오류와 파라미터 오류, 관측 시점을 구분해야 하며 전체 observer의 보편적 우월성을 뜻하지 않는다.',
'교차검증은 참가자 수준에서 분리한 내부 검증이며 새로운 외부 연구가 아니다. 다른 test 구성을 가진 6:2:2, 7:1:2, 8:1:1, CV의 점수 크기만으로 어느 분할이 더 우수하다고 해석하지 않는다.','',
'## 검증과 재현','',
'- 과거 PopPK export에는 cutoff_time 열이 일부 비어 있었다. first-k의 정의에 따라 이미 저장된 마지막 입력 관측 시각 support_end_time으로 해당 메타데이터를 채우고 원래 열을 별도로 보존했다. 입력·실제값·예측값·점수는 바꾸지 않았다 (`metadata_normalization.json`).',
'- 예측 파일 checksum, query/실제값 일치, drug-aware key 유일성, log/원농도 finite, 10 seeds를 검사했다.',
'- 재사용 seed 1의 k=3 파일 21개에서 기존 checksum 기록 불일치를 발견했다. 기존 파일·manifest를 보존하고 40개 조건을 독립 재추론했다. 현재 기존 파일과 예측값·실제값·ID·시각이 모두 일치했으며 예측 최대 차이는 0.0이다. 최종 집계는 새 검증 폴더를 사용한다 (`checksum_reconciliation.json`). 과거 byte-level 불일치의 발생 경위 자체를 복원했다고 주장하지 않는다.',
'- 240-epoch 실행의 120-epoch까지 선택된 가중치를 원래 체크포인트와 직접 비교했다 (`budget120_replay_checks.csv`).',
'- 초기 합성 simulation의 timing/noise/route 결합을 발견해 독립적으로 요인을 추출한 `simulation_crossed`를 추가했다. 초기 실행을 삭제하지 않았으며 기전 결과는 교차된 설계를 사용한다.',
'- raw data, participant/query별 자료, checkpoint는 서버에 보존한다. 공개용 자료 묶음에는 집계표·그림·코드·설정·원고만 포함한다.',
'- 독립 외부 cohort, 원 논문 비공개 자료의 성능 재현, 임상적 유효성은 이번 작업의 완료 범위가 아니다.','']
(S/'REPORT_KO.md').write_text('\n'.join(lines))
print(S/'REPORT_KO.md')
