# Entry points

Use the root README for the current experiment paths. Scripts retain the registered historical run/config IDs so restored artifacts resolve exactly. New publication displays use MoDE-PK. Training scripts must not create new random splits implicitly. The analysis entry points require their documented original results; source-only distribution is not a data bundle.

## Current primary-model analyses

Use `run_hybrid_primary.py` for the adopted molecular+identity model and `prepare_hybrid_primary.py` / `summarize_hybrid_primary.py` for its predictions. All earlier analysis runners below describe the archived molecular-only model unless they explicitly identify the hybrid primary. Restored parent artifacts are required; do not run the historical scripts as the current model by default.

## Archived molecular-only robustness experiments

- `run_completion_suite.py`: registered 240-epoch extensions (360 new fits; existing seed-1 runs reused).
- `run_completion_controls.py`: LSTM, MLP, prefix pooling, attention, and training-only synthetic augmentation.
- `run_completion_simulation_crossed.py`: independent-factor synthetic mechanism experiment; `run_completion_simulation.py` preserves the preliminary linked-factor design.
- `run_completion_centered_augmentation.py`: training-only generator-bias control using exactly the saved perturbations and noise.
- `analyze_completion_primary.py`: drug/horizon diagnostics using the same pooled fits as the primary paper tables.
- `analyze_completion_stages.py`: same-checkpoint parameter/state pathway diagnostics, distinct from retrained ablations.
- `verify_completion_reuse.py`: existing resampling and molecular-encoder prediction audits.
- `summarize_completion_suite.py`: validates all registered fits, queries, and checkpoint parity before aggregation.

These entry points require the registered original dataset, splits, calibration artifacts, and previous run assets at the paths specified in their code. No clinical data or experiment outputs are included in this Git directory. Do not launch a second scheduler over running jobs. The resume helper preserves running fits and limits new launches according to shared-server memory.
