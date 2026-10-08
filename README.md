# MoDA-PK

**MoDA-PK: Molecularly Informed Drug Adaptation for Pharmacokinetic Forecasting**

MoDA-PK forecasts individual drug concentration–time profiles from molecular structure, drug identity, dosing records, and available concentration measurements. It combines a drug-conditioned pharmacokinetic prior with observation-based adaptation and analytic compartment dynamics.

## Model overview

- Frozen 512-dimensional molecular representations and learned 512-dimensional drug-identity embeddings are concatenated and projected into a shared 32-dimensional drug context.
- The drug context establishes five effective pharmacokinetic parameters: absorption rate, clearance, central volume, intercompartmental clearance, and peripheral volume.
- A bidirectional GRU (BiGRU) encodes the concentration history available at each update. A dose-conditioned controller uses this representation to update effective clearance, central volume, and central drug amount.
- Future concentrations are predicted by propagating the compartment dynamics under recorded oral doses, intravenous boluses, and finite infusions.

## Repository contents

- `src/mhap_pk/`: model implementations, PK solver, training, data processing, and evaluation.
- `src/pkforecast/`: implementation dependencies.
- `scripts/`: training, experiment runners, evaluation, and figure-generation code.
- `configs/`: model and experiment configurations.
- `tests/`: numerical, metric, model, and calibration checks.
- `environment/`: dependency information, upstream versions, and required input hashes.

The repository contains implementation and analysis code. Manuscripts, result tables and figures, participant-level data, checkpoints, and run logs are not included.

## Installation

Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Download the pinned comparator sources and run the tests:

```bash
python scripts/fetch_baseline_sources.py
python -m pytest -q
```

Downloaded comparator sources are stored in `third_party/` and remain subject to their original licenses and usage terms.

## Data preparation

The study uses publicly available human pharmacokinetic concentration–time records covering 11 drugs. The underlying clinical records and processed participant-level inputs are not redistributed in this repository, and access follows the conditions of the respective data providers.

Place authorized inputs at the paths specified in `environment/required_inputs.json`, then check their availability and hashes:

```bash
python scripts/check_inputs.py
```

The check reports missing files until the required inputs have been supplied. Training and analysis also require the configuration files, split assignments, and other inputs expected by the selected workflow.

## Training

After preparing the required inputs, run MoDA-PK with a new output directory:

```bash
python scripts/train.py --fold known --seed 1 --model molecular_identity --output results/runs/reproduction_seed1
```

The `molecular_identity` option selects MoDA-PK. The training wrapper estimates population initialization from training profiles, selects checkpoints using validation data, and refuses to overwrite an existing output directory. Reproduction of a reported experiment requires its original split assignments and configuration.

## Evaluation and analysis

The manuscript evaluates new participants receiving drugs represented during training. The learning settings include single-drug and integrated full-data training, integrated training with limited target-drug data, and a matched target-only low-data comparison.

| Script | Purpose |
| --- | --- |
| `scripts/run_molecular_identity_validation.py` | MoDA-PK fits across configured folds and seeds |
| `scripts/run_hybrid_primary.py` | Training-data-budget, participant-partition, and single-drug comparisons |
| `scripts/prepare_hybrid_primary.py` | Assemble prediction results for analysis |
| `scripts/summarize_hybrid_primary.py` | Summarize primary and supporting evaluations |

Aggregation and plotting scripts require their corresponding predictions or checkpoints. Check each script's configured input and output paths before execution.
