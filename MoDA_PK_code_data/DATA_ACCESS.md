# Data access

Git에 포함되는 `data/`는 약물/seed/cohort 수준의 집계 결과입니다. 환자 ID, profile ID, observation ID, 시간별 개인 농도·예측을 포함하는 테이블은 Git에서 제외합니다.

`local_data/`는 기존 연구자의 로컬 분석을 위해 복사한 prediction 자료이며, 공개 데이터 release를 뜻하지 않습니다. 로컬 폴더가 없는 clone에서는 집계 결과 재현만 가능합니다.

행 단위 mixed cohort 전체의 재배포 권한은 확보됐다고 확인되지 않았습니다. 특히 기존 감사에서는 Monolix example 자료의 명시적인 row-level redistribution grant를 확인하지 못했습니다. [기존 접근 정책](docs/original_data_access.md)을 보존했습니다. 해당 정책과 과거 audit은 모든 11개 source에 대한 완전한 법적 허가 목록은 아닙니다.

허가된 원자료에서 재학습하려면 source별 출처·조건을 확인하고 원래 단위 변환, participant split, dose events, frozen molecular cache, 훈련-only scaling 및 calibration을 준비해야 합니다. 현재 원고의 Supplementary Table S1은 source 문헌과 링크를 제공합니다. 보편적인 자동 downloader는 제공하지 않습니다.
