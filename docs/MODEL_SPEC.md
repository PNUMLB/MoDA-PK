# MoDE-PK primary model

The current model combines molecular structure and drug identity in one shared forecasting backbone. `molecular_identity` is the training CLI name. Earlier `molecular_relation` checkpoints are a different model and must not be relabeled as current Ours.

1. Frozen Uni-Mol vector: 512 dimensions, standardized using training drugs only.
2. Learned drug-identity table: 512 dimensions per training drug. Unseen drugs use the mean training-table row and retain their own molecular vector. No test-drug entry is learned.
3. Concatenated 1024-dimensional input → Linear/SiLU → 32-dimensional shared context → five effective log-PK parameters. The shared base starts from training-only population calibration; the residual head starts at zero.
4. Dose-event analytic compartment dynamics propagate depot, central and peripheral amounts. Oral doses enter the depot; IV boluses and finite infusions enter the central compartment.
5. A 32-dimensional GRU reads 11 observation/event features plus the shared context. It outputs bounded prior-relative log-CL/log-Vc offsets and a sigmoid central-amount gain. Both update pathways are active.
6. The last parameters and updated state forecast future concentration under recorded doses. No future concentration enters inference.

The primary loss uses source/drug/participant-balanced query squared log error, parameter-offset penalty 0.001 and prior-deviation penalty 0.0001. Chemical and PK relation losses are OFF. AdamW: learning rate 0.003, weight decay 0.0001, gradient clipping 5. Validation at k=2 selects from epochs 0,10,...,120. Training history counts are 0,1,2,4; k=3 is additional inference. Seeds 1–10 are evaluated separately.

Known-drug splits group linked participants; joint drug–source folds exclude the target source/drugs from training and validation. Future queries are fixed after the fourth distinct observation time for all k. Clinical metadata and source IDs define grouping, not additional patient inputs.

Model entry: `src/mhap_pk/models/priors/molecular_identity.py`. Retrained controls: `hybrid_controls.py`. Shared dynamics/observer: `relational_model.py`. Dataset and split remain the registered source-traceable cohort. The model name does not imply different output heads per drug: one shared projection, parameter head and observer use distinct identity embeddings.

This model revision followed internal result inspection. It is not untouched external validation. The previous model, identity-only references, and failed alternatives retain separate identities. Removed-input, removed-update, and optional-loss conditions must be retrained, not inferred by relabeling old results. Same-checkpoint interventions are separate diagnostics.
