#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/utkface_groupdro_training.py"
ERM_DIR="$(cd "${SCRIPT_DIR}/../ERM" && pwd)"

GPU_ID="${GPU_ID:-}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
DATA_ROOT="${DATA_ROOT:-./data/UTKFace}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/utkface_groupdro_outputs}"
ERM_HPARAMS_DIR="${ERM_HPARAMS_DIR:-${ERM_DIR}/utkface_outputs/finetune}"
SEEDS_FILE="${SEEDS_FILE:-${OUTPUT_DIR}/experiment_seeds.json}"

BACKBONE="${BACKBONE:-vit_b_16}"
SEEDS="${SEEDS:-random}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-5}"
DEMO_MODES="${DEMO_MODES:-age,race}"
SELECTION_METRIC="${SELECTION_METRIC:-overall_acc}"
BASELINE_SELECTION_ATTR="${BASELINE_SELECTION_ATTR:-race}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROBUST_STEP_SIZE="${ROBUST_STEP_SIZE:-0.01}"
GAMMA="${GAMMA:-0.1}"
GEN_ADJ="${GEN_ADJ:-0.0}"

LR_GRID="${LR_GRID:-1e-2,1e-3,1e-4,5e-5,2e-5}"
WD_GRID="${WD_GRID:-1e-4,1e-5}"
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-32,64,128,256}"
ROBUST_STEP_GRID="${ROBUST_STEP_GRID:-0.001,0.01,0.1}"
GAMMA_GRID="${GAMMA_GRID:-0.01,0.1,1.0}"

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
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --robust-step-size "${ROBUST_STEP_SIZE}"
  --gamma "${GAMMA}"
  --generalization-adjustment "${GEN_ADJ}"
  --lr-grid "${LR_GRID}"
  --weight-decay-grid "${WD_GRID}"
  --batch-size-grid "${BATCH_SIZE_GRID}"
  --robust-step-size-grid "${ROBUST_STEP_GRID}"
  --gamma-grid "${GAMMA_GRID}"
  --selection-metric "${SELECTION_METRIC}"
  --baseline-selection-attr "${BASELINE_SELECTION_ATTR}"
)

if [[ "${SKIP_FINETUNE}" == "1" ]]; then
  common_args+=(--skip-finetune)
fi
if [[ "${FINETUNE_ONLY}" == "1" ]]; then
  common_args+=(--finetune-only)
fi
if [[ "${SKIP_BASELINE}" == "1" ]]; then
  common_args+=(--skip-baseline)
fi

echo "========== UTKFace GroupDRO (ERM hparams + shared seeds; ACC-based) =========="
echo "Output: ${OUTPUT_DIR}"
echo "Seeds file: ${SEEDS_FILE}"
echo "ERM hparams: ${ERM_HPARAMS_DIR}"
echo "Backbone: ${BACKBONE}"
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
    age) attrs="age"; sel="age" ;;
    race) attrs="race"; sel="race" ;;
    age_race|age,race) attrs="age,race"; sel="age" ;;
    *)
      echo "ERROR: unknown DEMO_MODE='${mode}' (use age|race|age_race)" >&2
      exit 1
      ;;
  esac
  echo ""
  echo "########## GroupDRO | demographic_attrs=${attrs} ##########"
  python3 "${TRAIN_SCRIPT}" \
    "${common_args[@]}" \
    --skip-baseline \
    --demographic-attrs "${attrs}" \
    --selection-attr "${sel}"
done

echo ""
echo "Done."
echo "  Seeds:   ${SEEDS_FILE}"
echo "  Results: ${OUTPUT_DIR}/results/"
echo "  Ckpts:   ${OUTPUT_DIR}/checkpoints/"
