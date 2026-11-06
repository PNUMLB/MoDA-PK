# MoDA-PK — revised code and data

**MoDA-PK: Molecularly Informed Drug Adaptation for Pharmacokinetic Forecasting**의 현재 수정 원고에 사용한 모델 코드, 집계 결과 데이터, 표·그림 생성 코드를 정리한 저장소입니다.

기준은 **원본 11개 약물**, **seed 1–5**, **원래 120-update full-batch 프로토콜**입니다. Propofol을 포함한 12-drug 장기학습, 경로별 공유, mini-batch 탐색 실험을 현재 핵심 결과와 섞지 않았습니다. 원고·ZIP·학습 checkpoint·개발 로그는 Git 대상에서 제외했습니다.

## 빠른 시작

Python 3.10 이상을 사용합니다. 검증에 사용한 버전은 `requirements-tested.txt`에 기록했습니다. 이 파일의 torch build 표기는 사용 환경에 따라 해당 PyTorch wheel index가 필요할 수 있으므로 기본 설치는 `pyproject.toml` 의존성을 사용합니다. 저장소 루트에서 실행하세요.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python examples/model_forward.py
python analysis/reproduce.py
```

- `model_forward.py`: 임상 자료 없이 synthetic tensor로 모델 import·출력 크기·finite prediction·파라미터 수를 확인합니다.
- `reproduce.py`: Git에 포함된 **집계 데이터만**으로 주요 표와 지표·ablation 그래프를 만듭니다. 재학습이나 GPU가 필요하지 않습니다.
- 출력은 Git에서 제외되는 `outputs/`에 생성됩니다.

## 포함한 코드와 데이터

```text
src/
  moda_pk.py              # FULL/ablation 생성 API
  moda_training.py        # 사전 처리된 tensor용 원래 full-batch 학습 loop
  mhap_pk/                # 실제 FULL 구현 및 import에 필요한 의존 모듈
analysis/
  reproduce.py            # 공개 대상 집계 데이터 → 수치표/LaTeX/그래프
  generate_full.py        # 로컬 prediction 자료 → PK 곡선 포함 전체 분석
scripts/train.py          # 허가된 사전 처리 tensor로 학습하는 진입점
examples/model_forward.py # Synthetic forward smoke test
configs/revised.json      # 원고 설정·차원·seed·ablation matrix
data/
  primary/                # 약물별/seed별 또는 cohort별 집계 지표
  essential/              # 5-seed matched ablation 및 transfer 집계 결과
  dose_controller/        # 71D→64D gate 대조시험; 주 모델에는 채택하지 않음
  model_source_sha256.json
docs/
  model.md                # 모델 flow, 차원, 수식, 파라미터 수 상세 설명
  original_data_access.md # 기존 데이터 접근·공개 범위 정책
  licenses/               # 포함된 의존 구현의 라이선스 사본
local_data/              # 로컬 prediction 재현 자료 — Git에서 제외
```

`src/mhap_pk`는 기존 구현의 dependency closure를 복사한 것입니다. BiGRU가 import하는 대안 observer 때문에 Mamba 참고 구현도 포함되지만, 현재 FULL은 **BiGRU**를 사용합니다. 관련 attribution은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 확인하세요.

## 모델 API와 차원

```python
import torch
from moda_pk import build_model

population = torch.log(torch.tensor([1., 1., 10., 1., 10.], dtype=torch.float64))
model = build_model(11, population)
```

위 초기값은 API 설명용 synthetic 값입니다. 논문 재학습에는 해당 훈련 자료만으로 calibration한 초기값을 사용해야 합니다.

- Frozen Uni-Mol 512D + trainable Drug-ID 512D → joint projection **1024→32**
- 32D drug context → effective log-PK residual **32→5**
- 11D 관측/이벤트 특성 + 32D context → **43D** history token
- Prefix BiGRU: 방향별 hidden 32D → concatenate 64D → projection **64→32**
- Parameter head **32→2**, amount-gain head **32→1**
- Dose controller: history32 + context32 + dose7 = **71→16→2**
- Depot/central/peripheral amount 상태 **3D** → 미래 log 농도

11-drug FULL의 학습 파라미터는 **56,751개**입니다. Frozen Uni-Mol encoder는 포함하지 않습니다. 현재 패키지는 Uni-Mol 사전학습 encoder나 분자 embedding cache를 배포하지 않습니다.

## 학습 데이터 접근과 학습 실행

집계 지표만으로 모델을 재학습할 수는 없습니다. 원래 훈련에는 행 단위 농도·투여 기록, 고정 participant split, frozen molecular cache, 훈련-only scaling 및 population calibration이 필요합니다. 이 저장소는 이를 자동 다운로드·복원하는 완전한 전처리 패키지는 아닙니다.

학습 loop는 허가된 자료로 별도 준비한 tensor payload를 입력받습니다.

```bash
python scripts/train.py \
  --inputs local_data/training_inputs.pt \
  --variant FULL --seed 1 --output outputs/train/full_s1
```

Payload 사양은 [docs/training_inputs.md](docs/training_inputs.md)에 있습니다. checkpoint 선택은 validation으로만 하며, 이 명령은 test inference를 수행하지 않습니다. 저장된 과거 실험의 모든 checkpoint에 대해 byte-identical replay를 보장하는 독립 재학습 release라고 주장하지 않습니다.

## 전체 그림 재생성 — 로컬 자료가 있는 경우

기존 workspace에서 만든 이 폴더에는 `local_data/MoDA_PK_results/`도 로컬용으로 배치했습니다. 이 폴더는 **Git commit에 포함되지 않습니다.** 집계 자료에는 환자별 곡선을 복원할 정보가 없으므로, Git clone만으로 PK 곡선 Figure 3–5를 생성할 수는 없습니다.

```bash
python -m pip install -e '.[full-analysis]'
python analysis/generate_full.py \
  --results local_data/MoDA_PK_results \
  --output outputs/full
```

- 현재 원고의 Table 1–5, 보충 집계표, Figure 3–8 및 보충 ablation 그림을 생성합니다.
- Figure 1·2는 데이터 그래프가 아닌 개념도이므로 원고의 벡터 원본을 사용합니다.
- 보충 S1·S2의 출처·전처리 적격성 메타데이터는 집계 예측에서 역산할 수 없습니다.
- 배치·폰트는 원래 PDF와 픽셀 단위로 동일하지 않을 수 있습니다.
- 원고에서 제외한 transfer 그림은 `--include-transfer`로 별도 생성합니다.
- Dose controller 64D 대조시험은 `--include-dose-controller-test`로 생성합니다.

## 평가 조건과 결과 해석

Train/validation/test는 **285/96/101 profiles**, **202/67/73 source-scoped participants**입니다.

학습 history는 `k={0,1,2,4}`, 학습·검증 query는 네 번째 관측 블록 이후, 검증 history는 k=2입니다. 주 test는 k=3 및 세 번째 블록 이후 관측입니다. 이 예측창 불일치는 보존된 한계입니다. 미래 농도는 feature로 사용하지 않으며, 미래 recorded dose는 기존 조건대로 허용합니다.

지표는 참가자별 query log-MSE를 평균한 뒤 drug RMSE를 계산하고, drug를 source 내부에서 평균한 다음 source를 평균하는 **SB-hRMSE**입니다. 표의 SD는 다섯 학습 seed의 sample SD이며 환자 모집단의 95% CI가 아닙니다.

| Variant | SB-hRMSE, mean ± SD |
|---|---:|
| FULL | 0.467 ± 0.011 |
| NO-UPDATE | 0.562 ± 0.055 |
| PARAM-ONLY | 0.572 ± 0.050 |
| AMOUNT-ONLY | 0.508 ± 0.013 |
| ID-ONLY | 0.520 ± 0.006 |
| MOL-ONLY | 0.509 ± 0.017 |

Full checkpoints 5개를 재사용했고, 5개 ablation을 각각 5회 독립 학습했습니다. 약물·seed별 수치는 `data/essential/`에 있습니다. Factorial interaction은 기술적 오차 지표이며 biological synergy를 뜻하지 않습니다.

Matched low-data pooling은 5개 약물에서 개선·6개에서 악화됐습니다. 별도 64D gate 시험은 평균 0.496 ± 0.010으로, 기존 71D gate보다 높았습니다. 현재 모델은 71D gate를 유지합니다.

현재 결과는 represented-drug 내부 평가입니다. 테스트 결과를 개발 중 확인한 이력이 있으므로 untouched external validation, unseen-drug generalization, physiological CL/Vc identification 또는 임상적 유용성을 주장하지 않습니다.

## Git commit 범위

`.gitignore`가 다음을 제외합니다.

- `local_data/`, raw/processed 행 단위 자료, Parquet/NPZ
- checkpoint/PT, ZIP, 생성 그림·표와 실행 결과
- 가상환경, Python cache, notebook checkpoint

약물·seed 수준 집계 CSV와 코드·설명만 기본 commit 대상입니다. 기존 데이터 공개 정책에 따르면 일부 source의 row-level 재배포 권한이 확인되지 않았습니다. `.gitignore`를 해제해서 전체 환자 예측을 올리는 방식으로 이 경계를 우회하지 마세요. [DATA_ACCESS.md](DATA_ACCESS.md)에 설명했습니다.

새 원격 저장소에 연결하려면 이 폴더 안에서 실행합니다.

```bash
git status --short
git add README.md DATA_ACCESS.md THIRD_PARTY_NOTICES.md .gitignore pyproject.toml \
  src analysis scripts examples configs data docs requirements-tested.txt
# 필요하면 tests도 추가
git diff --cached --stat
git commit -m "Add MoDA-PK revised model and reproducible aggregate analyses"
```

이 작업에서는 원격 push나 실제 commit은 수행하지 않습니다. 원고의 Data and code availability에 원격 주소·release tag를 확정한 뒤 연결하세요.

## 라이선스와 출처

원자료·pretrained representation·참고 코드의 upstream 조건은 각각 적용됩니다. 이 패키지가 제3자 자료의 재배포 권한을 새로 부여하지 않습니다. 프로젝트 전체에 대한 새 MIT/Apache 라이선스는 임의로 선언하지 않았습니다. 공개 라이선스는 권리자가 결정해야 합니다.
