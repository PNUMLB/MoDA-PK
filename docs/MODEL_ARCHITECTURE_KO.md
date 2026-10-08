# MoDE-PK 현재 모델 구조

확정 구성: **Uni-Mol + Drug-ID 결합 / parameter·state update 모두 사용 / relation loss OFF**.

## 입력과 약물 표현

고정 Uni-Mol의 512차원 분자 벡터와 학습 가능한 512차원 Drug-ID 벡터를 연결한다. 1024→32 Linear/SiLU가 하나의 공통 drug context를 만든다. 약물마다 별도의 예측 head를 두는 방식이 아니다. 공유 head가 이 context로 5개 effective log-PK parameter를 출력한다.

학습에 없는 약물은 자신의 분자 벡터를 그대로 사용하고, ID는 학습 약물 embedding의 평균으로 대체한다. 새 약물의 실제 농도 전체로 ID나 population prior를 추정하지 않는다. 약물 ID 외 source/participant는 모델 특징이 아닌 분할·집계 정보다.

## 투약과 관측

Oral은 흡수 depot, IV bolus는 중앙 compartment의 순간 투여, infusion은 일정 구간 동안 중앙 compartment로 유입되는 투여다. 투여량·시간·경로·지속시간이 analytic solver의 event를 정의한다.

과거 농도마다 11개 관측·투약 특징과 32차원 drug context를 GRU가 읽는다. 관측마다 CL·Vc의 bounded offset과 중앙 약물량을 보정한다. Hidden state와 약물량 state는 다르다. 마지막 관측 이후에는 최종 파라미터와 state에서 미래 투약을 반영하며 농도를 예측한다. Query 농도는 입력이 아니다.

## 학습과 평가

주 손실은 source→drug→participant 균형의 로그 농도 제곱오차다. PK parameter offset 및 prior deviation penalty만 추가한다. Chemical/PK relation은 현재 Ours에서 꺼져 있으며 별도 재학습 ablation이다. Prior와 observer는 함께 학습하며, Uni-Mol encoder는 고정이다.

Known은 이미 학습에 등장한 약물의 새 참가자, Joint는 학습과 validation에서 제외한 약물·source의 예측이다. k는 제공한 distinct 농도 관측 시점 수다. k=0,1,2,3,4 모두 네 번째 관측 이후의 동일 query로 평가한다. 실험은 seed 1–10을 별도로 학습·평가한다.

주 성능표는 약물별 participant-balanced error를 먼저 제시한다. 평균 점수, seed 평균 curve, 같은 checkpoint의 update 비활성화, 각각 재학습한 ablation은 서로 다른 분석이다. 기존 molecular-only 결과를 새 결합 모델의 결과로 사용하지 않는다.

영문 수치·구현 명세: [MODEL_SPEC.md](MODEL_SPEC.md). 기존 Information observer와 frozen-prior 탐색은 현 최종 모델이 아니다.
