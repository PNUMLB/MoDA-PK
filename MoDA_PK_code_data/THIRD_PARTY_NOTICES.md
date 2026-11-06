# Third-party notices

MoDA-PK 실행 경로의 의존 모듈 중 `mamba_observer.py`는 `state-spaces/mamba`의 Mamba-1 recurrence와 initialization을 참고한 구현입니다.

- Upstream: https://github.com/state-spaces/mamba
- Authors: Tri Dao and Albert Gu, Copyright 2023.
- License: Apache License 2.0; 사본은 `docs/licenses/MAMBA_APACHE_2_0.txt`입니다.
- Project changes: differentiable functional state updates, masking/reset, 43→32 input projection and float64 recurrence.
- 현재 FULL은 Mamba가 아니라 BiGRU를 사용합니다. 이 파일은 기존 import dependency 때문에 포함됩니다.

다른 데이터 제공자와 Uni-Mol pretrained representation의 조건은 별도입니다. 원자료와 frozen Uni-Mol encoder/cache는 Git 대상에 포함하지 않습니다. 이 저장소 전체에 대한 별도 프로젝트 공개 라이선스를 임의로 부여하지 않았습니다.
