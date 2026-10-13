#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/fairvision_erm_training.py"

GPU_ID="${GPU_ID:-}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
DATA_ROOT="${DATA_ROOT:-./data/FairVision}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/fairvision_outputs}"
SEEDS_FILE="${SEEDS_FILE:-${OUTPUT_DIR}/experiment_seeds.json}"

MODELS="${MODELS:-vgg16}"
SEEDS="${SEEDS:-random}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-5}"
DEMO_MODES="${DEMO_MODES:-gender,race}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
LR_BACKBONE="${LR_BACKBONE:-0}"
LR_HEAD="${LR_HEAD:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_AMP="${USE_AMP:-0}"
DEMO_EMBED_DIM="${DEMO_EMBED_DIM:-16}"
LR_GRID="${LR_GRID:-1e-2,1e-3,1e-4,5e-5,2e-5}"
WD_GRID="${WD_GRID:-1e-4,1e-5}"
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-32,64,128,256}"
DEMO_DIM_GRID="${DEMO_DIM_GRID:-8,16,32}"

SKIP_IMAGE_ONLY="${SKIP_IMAGE_ONLY:-0}"
SKIP_DEMO="${SKIP_DEMO:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
FINETUNE_ONLY="${FINETUNE_ONLY:-0}"

if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
mkdir -p "${TORCH_HOME}/hub/checkpoints" "${OUTPUT_DIR}"
export TORCH_HOME

common_args=(
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --seeds-file "${SEEDS_FILE}"
  --models "${MODELS}"
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

echo "========== FairVision ERM (finetune → multi-seed) =========="
echo "Output: ${OUTPUT_DIR}"
echo "Seeds file: ${SEEDS_FILE}"
echo "Models: ${MODELS} | epochs=${EPOCHS}"
echo "Finetune grids: lr=[${LR_GRID}] wd=[${WD_GRID}] batch_size=[${BATCH_SIZE_GRID}] demo_dim=[${DEMO_DIM_GRID}]"
echo "Demo modes: ${DEMO_MODES}"
echo "ERM ckpt metric: worst_subgroup"
echo "ERM+demo ckpt metric: min_improvement vs ERM"

if [[ "${SKIP_IMAGE_ONLY}" != "1" ]]; then
  echo ""
  echo "########## Phase 1: image-only ERM (worst-subgroup ckpt) ##########"
  # selection-attr=gender by default for shared ERM; override with SELECTION_ATTR
  python3 "${TRAIN_SCRIPT}" \
    "${common_args[@]}" \
    --selection-attr "${SELECTION_ATTR:-gender}"
else
  echo "Skipping image-only ERM (SKIP_IMAGE_ONLY=1)"
fi

if [[ "${SKIP_DEMO}" != "1" ]]; then
  if [[ "${SEEDS}" == "random" ]]; then
    demo_seeds_arg="random"
  else
    demo_seeds_arg="${SEEDS}"
  fi

  IFS=',' read -r -a DEMO_LIST <<< "${DEMO_MODES}"
  for mode in "${DEMO_LIST[@]}"; do
    mode="$(echo "${mode}" | xargs)"
    [[ -z "${mode}" ]] && continue
    case "${mode}" in
      gender) attrs="gender"; sel="gender" ;;
      race) attrs="race"; sel="race" ;;
      gender_race|gender,race) attrs="gender,race"; sel="gender" ;;
      *)
        echo "ERROR: unknown DEMO_MODE='${mode}' (use gender|race|gender_race)" >&2
        exit 1
        ;;
    esac
    echo ""
    echo "########## Phase 2: naive demo ERM (${attrs}, min_improvement vs ERM) ##########"
    python3 "${TRAIN_SCRIPT}" \
      "${common_args[@]}" \
      --seeds "${demo_seeds_arg}" \
      --use-demographics \
      --demographic-attrs "${attrs}" \
      --selection-attr "${sel}"
  done
else
  echo "Skipping demographic ERM (SKIP_DEMO=1)"
fi

echo ""
echo "Done."
echo "  Seeds:     ${SEEDS_FILE}"
echo "  Finetune:  ${OUTPUT_DIR}/finetune/"
echo "  Results:   ${OUTPUT_DIR}/results/"
