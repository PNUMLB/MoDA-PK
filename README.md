# MoDA-PK

**Molecularly Informed Drug Adaptation for Pharmacokinetic Forecasting**

MoDA-PK predicts future drug concentrations from molecular representations, drug identity, recorded dosing events, and the concentration history available for an individual. It combines a drug-conditioned effective PK prior with observation-guided parameter and central-amount updates inside analytic compartment dynamics.

This repository contains the revised **11-drug model**, **five-seed aggregate results**, and **table/figure generation code**. The files are located directly at the repository root. The separate 12-drug/propofol extension and exploratory route-sharing and mini-batch studies are outside this release.

## Quick start

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python examples/model_forward.py
python analysis/reproduce.py
```

The synthetic example checks model construction, output shape, finite predictions, and parameter count. The analysis command reproduces numerical tables and data figures from the included aggregate CSV files; it requires neither retraining nor a GPU. Generated files are written to the ignored `outputs/` directory.

The versions used for the local verification are recorded in `requirements-tested.txt`. The recorded PyTorch build may require a platform-specific wheel index; `pyproject.toml` provides the default installation dependencies.

## Repository layout

```text
src/
  moda_pk.py                 # Model/ablation construction API
  moda_training.py           # Original full-batch fitting loop for prepared tensors
  mhap_pk/                   # FULL model and its implementation dependencies
analysis/
  reproduce.py               # Aggregate data -> tables and figures
  generate_full.py           # Local prediction data -> complete numerical analysis
scripts/train.py             # Training entry point for authorized prepared inputs
examples/model_forward.py    # Synthetic forward smoke test
configs/revised.json         # Architecture, split sizes, seeds, and protocol
data/
  primary/                   # Drug/seed/cohort-level benchmark summaries
  essential/                 # Five-seed component and conditioning ablations
  dose_controller/           # Optional 71D vs 64D controller comparison
  model_source_sha256.json   # Implementation snapshot hashes
docs/
  model.md                   # Detailed architecture and computation order
  training_inputs.md         # Training payload schema
  original_data_access.md    # Historical data-access policy
  licenses/                  # Notices for included implementation dependencies
```

`src/mhap_pk` contains the dependency closure used to construct the original FULL model. A Mamba reference module remains an import dependency of the observer controls, but **the reported FULL model uses a BiGRU**. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Model architecture

```python
import torch
from moda_pk import build_model

# Synthetic values for API illustration, not calibrated clinical parameters.
population = torch.log(torch.tensor([1., 1., 10., 1., 10.], dtype=torch.float64))
model = build_model(11, population)
```

The main dimensions are:

1. Frozen Uni-Mol **512D** + trainable Drug-ID **512D** -> **1024D** joint input.
2. Linear projection with SiLU -> **32D** drug context.
3. Context -> **5D** effective log-PK prior: log ka, log CL, log Vc, log Q, log Vp.
4. Observation/event features **11D** + drug context **32D** -> **43D** history tokens.
5. Available-prefix BiGRU: **32D per direction**, followed by **64D -> 32D** projection.
6. Parameter head **32 -> 2**; amount-gain head **32 -> 1**.
7. Dose controller: history32 + context32 + dose7 -> **71 -> 16 -> 2**.
8. Analytic depot/central/peripheral dynamics -> future concentrations.

The 11-drug FULL model has **56,751 trainable parameters**. This excludes the frozen Uni-Mol encoder. The encoder weights and molecular embedding cache are not distributed in this repository. See [the detailed model description](docs/model.md) for equations, tensor shapes, and implementation paths.

## Included results and evaluation protocol

The original cohort contains **11 drugs, 482 profiles, and 342 source-scoped participants**. Train/validation/test counts are **285/96/101 profiles** and **202/67/73 participants**.

Training uses histories of `k={0,1,2,4}` blocks and targets after the fourth observation block. Validation uses `k=2` with the same target window. Primary testing supplies `k=3` and evaluates measurements after the third block. This original prediction-window mismatch is retained and reported as a limitation. Future concentrations are excluded from model inputs; recorded future doses are allowed under the existing protocol.

The primary aggregate is **source-balanced hierarchical log-RMSE (SB-hRMSE)**: average query squared log-error within participant, average participant MSE within drug and take its square root, average drugs within source, then average sources. Sample SD across fitting seeds is not a patient-population 95% confidence interval.

| Variant | SB-hRMSE, mean ± sample SD over five seeds |
|---|---:|
| FULL | 0.467 ± 0.011 |
| NO-UPDATE | 0.562 ± 0.055 |
| PARAM-ONLY | 0.572 ± 0.050 |
| AMOUNT-ONLY | 0.508 ± 0.013 |
| ID-ONLY | 0.520 ± 0.006 |
| MOL-ONLY | 0.509 ± 0.017 |

FULL reuses the five original checkpoints. Each of the five controls was independently fitted for seeds 1–5: 25 ablation fits in total. Drug-level results and seed-paired differences are included in `data/essential/`. Factorial interaction is a descriptive error-scale quantity, not evidence of biological or causal synergy.

Matched two-participant target training showed improvements for five drugs and deterioration for six when other-drug records were added. The separate controller test yielded **0.496 ± 0.010** after removing the seven direct dose inputs, compared with **0.467 ± 0.011** for FULL; the main model retains the 71D controller.

These are internal evaluations for drugs represented during training. Prior test inspection during model development prevents an untouched external-validation claim. Unseen-drug generalization, physiological parameter identification, and clinical utility were not established.

## Training with authorized inputs

Aggregate scores cannot be used to retrain the forecasting model. Training requires authorized concentration/dose records, the original participant split, frozen molecular embeddings, training-only standardization, and population calibration. A universal raw-data downloader and complete mixed-cohort preprocessing pipeline are not provided.

The included fitter accepts an externally prepared tensor payload:

```bash
python scripts/train.py \
  --inputs local_data/training_inputs.pt \
  --variant FULL --seed 1 --output outputs/train/full_s1
```

See [docs/training_inputs.md](docs/training_inputs.md). Checkpoint selection uses validation only; this command does not perform test inference. This interface is not a claim of byte-identical end-to-end reconstruction of every historical fit.

## Full prediction-based figure generation

Participant-level prediction records are **not included in Git**. If you have an authorized local copy of the results package, place it at `local_data/MoDA_PK_results/` and run:

```bash
python -m pip install -e '.[full-analysis]'
python analysis/generate_full.py \
  --results local_data/MoDA_PK_results \
  --output outputs/full
```

This generates the primary numerical tables, PK-case figures, metric figures, and essential ablation figure. Figures 1 and 2 are conceptual diagrams supplied with the manuscript. Source/provenance and eligibility tables require metadata that cannot be recovered from prediction summaries. Regenerated layouts are not pixel-identical to the original manuscript; no unrecorded dense trajectories, dosing annotations, or uncertainty intervals are invented.

Optional diagnostics excluded from the current main manuscript:

- `--include-transfer`: plot the archived signed cross-drug pooling differences.
- `--include-dose-controller-test`: plot the 71D vs 64D controller comparison.

## Data access, licensing, and Git exclusions

Git includes code and drug/seed/cohort-level aggregate data. `.gitignore` excludes `local_data/`, participant-level Parquet/NPZ files, checkpoints, archives, generated outputs, and environment caches. A clone therefore supports aggregate reproduction, but not patient-specific PK-curve reproduction without separately authorized inputs.

Row-level redistribution permission for the complete mixed cohort has not been established. See [DATA_ACCESS.md](DATA_ACCESS.md). The source links and literature references are documented in Supplementary Table S1 of the manuscript. Do not remove the local-data exclusions to bypass those provider-specific conditions.

Upstream software, clinical data, and pretrained representations retain their respective terms. No new blanket license is granted to third-party material, and no repository-wide permissive license has been invented. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
