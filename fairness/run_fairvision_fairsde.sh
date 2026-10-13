#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/fairvision_fairsde_training.py"
ERM_DIR="$(cd "${SCRIPT_DIR}/../ERM" && pwd)"

GPU_ID="${GPU_ID:-}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
DATA_ROOT="${DATA_ROOT:-./data/FairVision}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/fairvision_fairsde_outputs}"
ERM_HPARAMS_DIR="${ERM_HPARAMS_DIR:-${ERM_DIR}/fairvision_outputs/finetune}"
SEEDS_FILE="${SEEDS_FILE:-${OUTPUT_DIR}/experiment_seeds.json}"

BACKBONE="${BACKBONE:-vgg16}"
SEEDS="${SEEDS:-random}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-5}"
DEMO_MODES="${DEMO_MODES:-gender,race}"
BASELINE_SELECTION_ATTR="${BASELINE_SELECTION_ATTR:-gender}"

EPOCHS="${EPOCHS:-20}"
ERM_EPOCHS="${ERM_EPOCHS:-20}"
SDE_EPOCHS="${SDE_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LAMBDA1="${LAMBDA1:-0.1}"
LAMBDA2="${LAMBDA2:-1.0}"
LAMBDA3="${LAMBDA3:-1.0}"
TEMPERATURE="${TEMPERATURE:-0.07}"
SELECTION_METRIC="${SELECTION_METRIC:-overall_auc}"
USE_AMP="${USE_AMP:-0}"

LR_GRID="${LR_GRID:-1e-2,1e-3,1e-4,5e-5,2e-5}"
WD_GRID="${WD_GRID:-1e-4,1e-5}"
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-32,64,128,256}"
LAMBDA1_GRID="${LAMBDA1_GRID:-0.01,0.1,1.0}"
TEMPERATURE_GRID="${TEMPERATURE_GRID:-0.07,0.1,0.2}"

SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
FINETUNE_ONLY="${FINETUNE_ONLY:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
BASELINE_ONLY="${BASELINE_ONLY:-0}"

if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
mkdir -p "${TORCH_HOME}/hub/checkpoints" "${OUTPUT_DIR}"
export TORCH_HOME

common_args=(
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --erm-hparams-dir "${ERM_HPARAMS_DIR}"
  --seeds-file "${SEEDS_FILE}"
  --backbone "${BACKBONE}"
  --seeds "${SEEDS}"
  --num-random-seeds "${NUM_RANDOM_SEEDS}"
  --epochs "${EPOCHS}"
  --erm-epochs "${ERM_EPOCHS}"
  --sde-epochs "${SDE_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --lambda1 "${LAMBDA1}"
  --lambda2 "${LAMBDA2}"
  --lambda3 "${LAMBDA3}"
  --temperature "${TEMPERATURE}"
  --selection-metric "${SELECTION_METRIC}"
  --lr-grid "${LR_GRID}"
  --weight-decay-grid "${WD_GRID}"
  --batch-size-grid "${BATCH_SIZE_GRID}"
  --lambda1-grid "${LAMBDA1_GRID}"
  --temperature-grid "${TEMPERATURE_GRID}"
  --baseline-selection-attr "${BASELINE_SELECTION_ATTR}"
)

if [[ "${USE_AMP}" == "1" ]]; then
  common_args+=(--use-amp)
else
  common_args+=(--no-use-amp)
fi

if [[ "${SKIP_FINETUNE}" == "1" ]]; then
  common_args+=(--skip-finetune)
fi
if [[ "${FINETUNE_ONLY}" == "1" ]]; then
  common_args+=(--finetune-only)
fi
if [[ "${SKIP_BASELINE}" == "1" ]]; then
  common_args+=(--skip-baseline)
fi

echo "========== FairVision FairSDE (ERM hparams + shared seeds) =========="
echo "Output: ${OUTPUT_DIR}"
echo "Seeds file: ${SEEDS_FILE}"
echo "ERM hparams: ${ERM_HPARAMS_DIR}"
echo "Backbone: ${BACKBONE} | ERM=${ERM_EPOCHS} SDE=${SDE_EPOCHS} | batch=${BATCH_SIZE}"
echo "Lambdas: λ1=${LAMBDA1} λ2=${LAMBDA2} λ3=${LAMBDA3} τ=${TEMPERATURE}"
echo "Selection metric: ${SELECTION_METRIC}"

echo ""
if [[ "${SELECTION_METRIC}" == "min_improvement" || "${BASELINE_ONLY}" == "1" ]]; then
  echo "########## Phase 0: image-only ERM baselines ##########"
  python3 "${TRAIN_SCRIPT}" \
    "${common_args[@]}" \
    --baseline-only

  if [[ "${BASELINE_ONLY}" == "1" ]]; then
    echo "BASELINE_ONLY=1; done after baselines."
    exit 0
  fi
else
  echo "Skipping Phase 0 baselines (SELECTION_METRIC=${SELECTION_METRIC})"
fi

IFS=',' read -r -a DEMO_LIST <<< "${DEMO_MODES}"
for mode in "${DEMO_LIST[@]}"; do
  mode="$(echo "${mode}" | xargs)"
  [[ -z "${mode}" ]] && continue
  case "${mode}" in
    gender) subgroup="gender"; sel="gender" ;;
    race) subgroup="race"; sel="race" ;;
    gender_race|gender,race) subgroup="gender_race"; sel="gender" ;;
    *)
      echo "ERROR: unknown DEMO_MODE='${mode}' (use gender|race|gender_race)" >&2
      exit 1
      ;;
  esac
  echo ""
  echo "########## FairSDE demographic_subgroup=${subgroup} ##########"
  python3 "${TRAIN_SCRIPT}" \
    "${common_args[@]}" \
    --skip-baseline \
    --demographic-subgroup "${subgroup}" \
    --selection-attr "${sel}"
done

echo ""
echo "Done."
echo "  Seeds:       ${SEEDS_FILE}"
echo "  Checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo "  Results:     ${OUTPUT_DIR}/results/"
