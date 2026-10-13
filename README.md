# ContextFair: Contextual Fairness Learning with Attribute Utility Regularization

## Abstract

Fairness learning aims to address model biases that favor certain subgroups while leaving others under-optimized. Many existing methods treat demographic attributes as spurious information to be removed, based on the assumption that a model without access to such information would treat samples equally across demographic groups. However, this would violate reality, especially in clinical domains, where demographic attributes can also be informative for prediction in many contexts; for example, age can offer meaningful diagnostic information for age-related diseases. Meanwhile, existing fairness learning paradigms that simply remove demographic information may adversely affect both overall and subgroup-specific predictive performance. Such approaches often equalize subgroup performance, but overlook disparities in performance gains across subgroups. To address these issues, we introduce ContextFair, a new context-aware fairness learning framework that considers sensitive attributes, such as demographic information as context, to improve overall predictive performance as well as fair gains across subgroups. It first computes a context utility score based on both the overall performance gain and the distribution of subgroup-specific improvements. The score then guides a neural modulation module that adaptively controls the use of demographic information when it benefits overall performance and promotes fair subgroup improvements while suppressing it when fairness is compromised. We further introduce a new fairness metric to quantify fair performance improvement across subgroups. Experiments on three real-world image datasets show that ContextFair consistently improves model fairness while maintaining strong overall performance.

---

Code for **ContextFair**, including all comparison models:

- **ERM** (image-only) and **ERM with demographic attributes**
- **Fusion** strategies (early / late / gated / FiLM)
- **Traditional fairness** methods (FairDRO, FairREAD, FairSDE, FairTune, GroupDRO, LAFTR)
- **ContextFair** (proposed)

Experiments cover three datasets: **CheXpert**, **FairVision (Glaucoma)**, and **UTKFace**. Each method folder provides one shell launcher per dataset (e.g. `run_chexpert_*.sh`, `run_fairvision_*.sh`, `run_utkface_*.sh`). Examples below use FairVision; the other datasets follow the same pattern.

---

## Repository layout

```
ContextFair_implementation_code/
├── requirements.txt
└── ContextFair_and_Comp_models/
    ├── datasets/       # CheXpert, FairVision, UTKFace loaders
    ├── ERM/            # Image-only ERM + naive demographic ERM
    ├── Fusion/         # Early / late / gated / FiLM fusion
    ├── fairness/       # FairDRO, FairREAD, FairSDE, FairTune, GroupDRO, LAFTR
    ├── contextfair/    # Proposed ContextFair model
    └── evaluation/     # Metrics from saved test predictions
```

---

## Setup

```bash
pip install -r requirements.txt
```

For GPU training, install a matching CUDA build of `torch` / `torchvision` from [pytorch.org](https://pytorch.org) if needed.

Point data roots via `DATA_ROOT` (defaults below assume relative paths under the working directory):

| Dataset    | Default `DATA_ROOT`   | Sensitive attributes (defaults) |
|------------|-----------------------|----------------------------------|
| CheXpert   | `./data/CheXpert`     | `sex`, `age`                     |
| FairVision | `./data/FairVision`   | `gender`, `race`                 |
| UTKFace    | `./data/UTKFace`      | `age`, `race` (target: gender)   |

---

## Finetune protocol and random seeds (all methods)

Executing any `bash run_*.sh` command, the script uses **5 random seeds by default** (`NUM_RANDOM_SEEDS=5`). It **automatically finetunes on one seed first** (the first of those seeds), then **reuses the selected hyperparameters for the remaining seeds**. This design is consistent across ERM, Fusion, fairness baselines, and ContextFair.

---

## Recommended run order

1. Train **ERM** first — Fusion, fairness baselines, and ContextFair can reuse ERM finetune summaries from `ERM/*/finetune/` via `ERM_HPARAMS_DIR`.
2. Run **Fusion**, **fairness**, and/or **ContextFair**.
3. Compute metrics with `evaluation/` on the saved `test_predictions.csv` files.

Override the GPU with `GPU_ID=0` (sets `CUDA_VISIBLE_DEVICES`). Most knobs are environment variables on the shell scripts.

---

## 1. ERM and ERM with demographics

**Location:** `ContextFair_and_Comp_models/ERM/`

Each launcher runs two phases:

1. **Image-only ERM**
2. **Naive demographic ERM** (`--use-demographics`) for each entry in `DEMO_MODES`

```bash
cd ContextFair_and_Comp_models/ERM
GPU_ID=0 bash run_fairvision_erm.sh
```

Useful flags: `SKIP_IMAGE_ONLY=1` (demographic ERM only), `SKIP_DEMO=1` (image-only only), `MODELS=vgg16`, `DEMO_MODES=gender,race`.

Outputs: `ERM/fairvision_outputs/{finetune,checkpoints,results}/`.

---

## 2. Fusion strategies

**Location:** `ContextFair_and_Comp_models/Fusion/`

Fusion methods: `early`, `late`, `gated`, `film` (default `FUSION_METHODS=all`).

```bash
cd ContextFair_and_Comp_models/Fusion
GPU_ID=0 bash run_fairvision_fusion.sh

# Optional: subset of methods / attributes
FUSION_METHODS=early,film DEMO_MODES=gender bash run_fairvision_fusion.sh
```
In **Phase 0**, the local image-only ERM baseline independently selects the random seeds and saves them in `experiment_seeds.json` within the Fusion output directory. The Fusion models then use the same saved seeds for training, enabling each Fusion run to be compared with its corresponding same-seed local ERM baseline. The initial ERM finetune summaries in `ERM_HPARAMS_DIR` are used only to reuse training hyperparameters, such as the learning rate, weight decay, batch size, etc. This design is consistent for **fairness and contextfair** models as well. 

---

## 3. Traditional fairness models

**Location:** `ContextFair_and_Comp_models/fairness/`

Supported methods: `fairdro`, `fairread`, `fairsde`, `fairtune`, `groupdro`, `laftr`.

```bash
cd ContextFair_and_Comp_models/fairness

# One method
GPU_ID=0 bash run_fairvision_fairdro.sh

# Full suite (or a subset)
GPU_ID=0 bash run_fairvision_fairness.sh
METHODS=groupdro,laftr DEMO_MODES=gender bash run_fairvision_fairness.sh
```

---

## 4. ContextFair (proposed)

**Location:** `ContextFair_and_Comp_models/contextfair/`

Staged training with attribute-utility regularization (Phase 0 ERM baselines → Stages 1–4).

```bash
cd ContextFair_and_Comp_models/contextfair
GPU_ID=0 bash run_fairvision_contextfair.sh

# Optional overrides
DEMO_MODES=gender BACKBONES=vgg16 FUSION_TYPE=film bash run_fairvision_contextfair.sh
```

Uses ERM finetune hyperparameters by default (`ERM_HPARAMS_DIR`). Outputs and checkpoints are written under `fairvision_contextfair_outputs/` and `checkpoints/` next to the scripts.

---

## 5. Evaluation metrics

**Location:** `ContextFair_and_Comp_models/evaluation/`

### What to evaluate

After training, each method writes per-seed CSVs under its results folder:

```text
<run_outputs>/results/<Method>_<attrs>/seed_<seed>/test_predictions.csv
```

Pass that **`<Method>_<attrs>` directory** (the parent of `seed_*`) to the evaluator—not an individual CSV.

| Variable | Meaning |
|----------|---------|
| `PRED` | Method to score (must contain `seed_*/test_predictions.csv`) |
| `BASE` | Optional image-only ERM results dir with the **same seeds** (enables Utility Score) |
| `DEMOGRAPHIC` | Attribute used for fairness / utility (`gender`, `race`, `sex`, `age`, …) |
| `OUT` | Where metrics are written (default: `$PRED/metrics`) |

Use `run_eval_fairvision.sh`, `run_eval_chexpert.sh`, or `run_eval_utkface.sh` to match the dataset. Paths may be relative to `evaluation/` (e.g. `../Fusion/...`) or absolute; do **not** use a bare `/Fusion/...` path from the filesystem root.

### Example (FairVision Fusion vs local ERM baseline)

```bash
cd ContextFair_and_Comp_models/evaluation

# Fairness metrics only
PRED=../Fusion/fairvision_fusion_outputs/results/filmFusion_gender \
DEMOGRAPHIC=gender \
bash run_eval_fairvision.sh

# Fairness + Utility Score (pair each seed with the local image-only ERM)
PRED=../Fusion/fairvision_fusion_outputs/results/filmFusion_gender \
BASE=../Fusion/fairvision_fusion_outputs/results/ERM \
DEMOGRAPHIC=gender \
bash run_eval_fairvision.sh
```

Set `DEMOGRAPHIC` to the attribute used in training (e.g. `gender` for `*_gender`, `race` for `*_race`).

Outputs land in `$PRED/metrics/` (CSVs, summary text, `eval_config.json`).

### Python entrypoint

```bash
python3 compute_metrics.py \
  --predictions-dir ../Fusion/fairvision_fusion_outputs/results/filmFusion_gender \
  --baseline-dir ../Fusion/fairvision_fusion_outputs/results/ERM \
  --demographic gender \
  --output-dir ../Fusion/fairvision_fusion_outputs/results/filmFusion_gender/metrics
```

---

## Common environment variables

| Variable | Meaning |
|----------|---------|
| `GPU_ID` | CUDA device id |
| `DATA_ROOT` | Dataset root |
| `OUTPUT_DIR` | Run output directory |
| `MODELS` / `BACKBONES` | Backbone(s), e.g. `resnet50`, `vgg16`, `vit_b_16` |
| `DEMO_MODES` | Sensitive attribute(s), comma-separated |
| `SEEDS` | `random` or comma-separated ints |
| `NUM_RANDOM_SEEDS` | Number of seeds when `SEEDS=random` (default 5) |
| `SKIP_FINETUNE` / `FINETUNE_ONLY` | Skip or only run hyperparameter search |
| `ERM_HPARAMS_DIR` | Path to ERM finetune summaries |

**See each `run_*.sh` for method-specific grids and defaults.**
