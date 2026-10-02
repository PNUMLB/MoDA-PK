# Model specification

**2026-09-23 implementation reference:** [MHAP-PK 모델 구조 상세 설명](MODEL_ARCHITECTURE_KO.md) covers the Sequential observer and the frozen shared-prior Information + state experiments. The subsequent [공동 추정 구현 명세](JOINT_ESTIMATION_KO.md) describes the new fixed-penalty joint estimator and interchangeable observation encoders, currently validated only in a small inner-development pilot.

The paragraph below describes the intended experimental design, not a declaration that every ablation has been completed.

The initial shared-prior comparison keeps the frozen molecular representation, small prior head, analytic dose-event compartment solver, and support-conditioned adapter. Parameter-only, state-only, and parameter+state GRU conditions are retrained from the same frozen prior. Restricted MAP is evaluated separately and is not accepted solely because it reduces support residual.
