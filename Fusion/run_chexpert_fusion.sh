#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/chexpert_fusion_training.py"
ERM_DIR="$(cd "${SCRIPT_DIR}/../ERM" && pwd)"

GPU_ID="${GPU_ID:-}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
DATA_ROOT="${DATA_ROOT:-./data/CheXpert}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/chexpert_fusion_outputs}"
ERM_HPARAMS_DIR="${ERM_HPARAMS_DIR:-${ERM_DIR}/chexpert_outputs/finetune}"
SEEDS_FILE="${SEEDS_FILE:-${OUTPUT_DIR}/experiment_seeds.json}"

MODELS="${MODELS:-resnet50}"
FUSION_METHODS="${FUSION_METHODS:-all}"
SEEDS="${SEEDS:-random}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-5}"
DEMO_MODES="${DEMO_MODES:-sex,age}"
BASELINE_SELECTION_ATTR="${BASELINE_SELECTION_ATTR:-sex}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
LR_BACKBONE="${LR_BACKBONE:-0}"
LR_HEAD="${LR_HEAD:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_AMP="${USE_AMP:-1}"
DEMO_EMBED_DIM="${DEMO_EMBED_DIM:-16}"
LR_GRID="${LR_GRID:-1e-2,1e-3,1e-4,5e-5,2e-5}"
WD_GRID="${WD_GRID:-1e-4,1e-5}"
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-32,64,128,256}"
DEMO_DIM_GRID="${DEMO_DIM_GRID:-8,16,32}"

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
  --models "${MODELS}"
  --fusion-methods "${FUSION_METHODS}"
  --seeds "${SEEDS}"
  --num-random-seeds "${NUM_RANDOM_SEEDS}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --lr-backbone "${LR_BACKBONE}"
  --lr-head "${LR_HEAD}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --demo-embed-dim "${DEMO_EMBED_DIM}"
  --lr-grid "${LR_GRID}"
  --weight-decay-grid "${WD_GRID}"
  --batch-size-grid "${BATCH_SIZE_GRID}"
  --demo-embed-dim-grid "${DEMO_DIM_GRID}"
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

echo "========== CheXpert Fusion (frontal-only; ERM hparams + Fusion shared seeds) =========="
echo "Output: ${OUTPUT_DIR}"
echo "ERM hparams: ${ERM_HPARAMS_DIR}  (hparams only; not ERM seeds)"
echo "Seeds file: ${SEEDS_FILE}  (shared by baseline ERM and fusion)"
echo "Models: ${MODELS} | fusion=${FUSION_METHODS} | epochs=${EPOCHS} | bs=${BATCH_SIZE}"
echo "Baseline: worst_subgroup_macro_auc (${BASELINE_SELECTION_ATTR})"
echo "Fusion:   min_macro_auc_improvement"
echo "Demo modes: ${DEMO_MODES}"

# Phase 0: train baselines once (same shared seeds as fusion) for all models.
echo ""
echo "########## Phase 0: image-only ERM baselines (ERM hparams + Fusion seeds) ##########"
python3 "${TRAIN_SCRIPT}" \
  "${common_args[@]}" \
  --baseline-only

if [[ "${BASELINE_ONLY}" == "1" ]]; then
  echo "BASELINE_ONLY=1; done after baselines."
  echo "  Seeds:      ${SEEDS_FILE}"
  echo "  Baselines:  ${OUTPUT_DIR}/checkpoints/ERM/"
  exit 0
fi

IFS=',' read -r -a DEMO_LIST <<< "${DEMO_MODES}"
for mode in "${DEMO_LIST[@]}"; do
  mode="$(echo "${mode}" | xargs)"
  [[ -z "${mode}" ]] && continue
  case "${mode}" in
    sex|gender) attrs="sex"; sel="sex" ;;
    age) attrs="age"; sel="age" ;;
    sex_age|sex,age|gender_age|gender,age) attrs="sex,age"; sel="sex" ;;
    *)
      echo "ERROR: unknown DEMO_MODE='${mode}' (use sex|gender|age|sex_age)" >&2
      exit 1
      ;;
  esac
  echo ""
  echo "########## Fusion demographics=${attrs} (selection_attr=${sel}) ##########"
  # Baselines already trained; skip re-training on subsequent demo modes.
  python3 "${TRAIN_SCRIPT}" \
    "${common_args[@]}" \
    --skip-baseline \
    --demographic-attrs "${attrs}" \
    --selection-attr "${sel}"
done

echo ""
echo "Done."
echo "  Seeds:     ${SEEDS_FILE}"
echo "  Baselines: ${OUTPUT_DIR}/checkpoints/ERM/"
echo "  Finetune:  ${OUTPUT_DIR}/finetune/"
echo "  Results:   ${OUTPUT_DIR}/results/"
