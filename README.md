# MoDE-PK

**Molecularly Conditioned Dose-Event Pharmacokinetics** combines molecular parameter prediction, analytic compartment dynamics with dosing events, and sequential concentration observation updates.

## Repository contents

- `src/mhap_pk/`: models, PK solver, training, data processing and evaluation. The import name is preserved for checkpoint compatibility.
- `src/pkforecast/`: implementation dependencies.
- `scripts/`: training, experiment runners, evaluation and analysis/figure generation code.
- `configs/`: research and experiment configuration.
- `tests/`: numerical, metric, model and calibration checks.
- `environment/`: dependencies, upstream versions and required input hashes.
- `docs/`: model/evaluation specifications, source links and repository scope.

Manuscripts, result tables/figures, patient-level data, checkpoints and run logs are not included. Analysis **code** is retained; analysis **results** are excluded. See `docs/REPOSITORY_SCOPE.md`.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/fetch_baseline_sources.py
python -m pytest -q
python scripts/check_inputs.py
```

The input check reports missing data until authorized inputs are restored. See `DATA_ACCESS.md` and `docs/DATASET_SOURCES.csv`. This is not a self-contained data bundle. Downloaded comparator sources go into ignored `third_party/`; their origin and usage notices remain in `THIRD_PARTY_NOTICES.md`.

## One training run

After restoring required inputs:

```bash
python scripts/train.py --fold known --seed 1 --model molecular_relation --output results/runs/reproduction_seed1
```

The historical condition key `molecular_relation` identifies MoDE-PK. The wrapper refuses an existing output directory, estimates population initialization from training profiles, and selects checkpoints on validation data. Do not regenerate an existing experiment split and call it the original split.

## Experiments

- `run_ten_seed_extension.py`: additional optimization seeds using registered conditions.
- `run_aligned_components.py`: component controls.
- `run_resampling_validation.py`: CV and alternative participant splits; a large experiment, not a smoke test.
- `run_scale_shape_diagnostic.py`: frozen-checkpoint calibration diagnostic. This is an experimental control, not the default MoDE-PK architecture.
- `summarize_*.py`, `analyze_*.py`, `plot_*.py`: aggregation and visualization after restoring their prediction/checkpoint dependencies. Some scripts expect historical run directories; inspect their input paths before execution.

Dataset redistribution permissions have not been verified for this release. No repository-wide software license is invented here. Literature comparator implementation scope is documented in source and notices; deterministic DCM is not ME-DCM.
