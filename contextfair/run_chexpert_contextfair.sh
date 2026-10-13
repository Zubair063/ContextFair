#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/chexpert_contextfair_model.py"
ERM_DIR="$(cd "${SCRIPT_DIR}/../ERM" && pwd)"

GPU_ID="${GPU_ID:-}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
DATA_ROOT="${DATA_ROOT:-./data/CheXpert}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/chexpert_contextfair_outputs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints}"
ERM_OUTPUT_DIR="${ERM_OUTPUT_DIR:-${ERM_DIR}/chexpert_outputs}"
ERM_HPARAMS_DIR="${ERM_HPARAMS_DIR:-${ERM_OUTPUT_DIR}/finetune}"
SEEDS_FILE="${SEEDS_FILE:-${OUTPUT_DIR}/experiment_seeds.json}"

SEEDS="${SEEDS:-random}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-5}"
DEMO_MODES="${DEMO_MODES:-sex,age}"
BACKBONES="${BACKBONES:-resnet50}"
TRAIN_DISEASES="${TRAIN_DISEASES:-13}"
FUSION_TYPE="${FUSION_TYPE:-film}"
DEVICE="${DEVICE:-cuda:0}"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
FEATURE_DIM="${FEATURE_DIM:-512}"
DEMO_EMBED_DIM="${DEMO_EMBED_DIM:-16}"
GAMMA="${GAMMA:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
RESIDUAL_BLOCKS="${RESIDUAL_BLOCKS:-1}"
HEAD_LAYERNORM="${HEAD_LAYERNORM:-1}"
PRETRAINED="${PRETRAINED:-1}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
STAGE4_EPOCHS="${STAGE4_EPOCHS:-20}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE2_LR="${STAGE2_LR:-1e-4}"
STAGE4_HEAD_LR="${STAGE4_HEAD_LR:-1e-4}"
STAGE4_ENCODER_LR="${STAGE4_ENCODER_LR:-2e-5}"
STAGE4_UNFREEZE_ENCODER="${STAGE4_UNFREEZE_ENCODER:-1}"
STAGE4_UNFREEZE_FROM_BLOCK="${STAGE4_UNFREEZE_FROM_BLOCK:-4}"
STAGE4_UNFREEZE_INCLUDE_FC="${STAGE4_UNFREEZE_INCLUDE_FC:-1}"
USE_COSINE_SCHEDULE_STAGE4="${USE_COSINE_SCHEDULE_STAGE4:-1}"
WARMUP_EPOCHS_STAGE4="${WARMUP_EPOCHS_STAGE4:-2}"
STAGE4_CHECKPOINT_SKIP_EPOCHS="${STAGE4_CHECKPOINT_SKIP_EPOCHS:-2}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.01}"

ALPHA="${ALPHA:-50.0}"
LAMBDA_CV="${LAMBDA_CV:-1}"
LAMBDA_IMP="${LAMBDA_IMP:-0.2}"
LAMBDA_WORST="${LAMBDA_WORST:-0.1}"
LAMBDA_KD="${LAMBDA_KD:-0.5}"
TEMPERATURE="${TEMPERATURE:-3.0}"
FAIRNESS_USE_RELU="${FAIRNESS_USE_RELU:-0}"

SELECTION_METRIC_STAGE1="${SELECTION_METRIC_STAGE1:-worst_subgroup}"
SELECTION_METRIC_STAGE2="${SELECTION_METRIC_STAGE2:-min_improvement}"
SELECTION_METRIC_STAGE4="${SELECTION_METRIC_STAGE4:-min_improvement}"
TIE_BREAK="${TIE_BREAK:-on}"
SKIP_ERM_HPARAMS="${SKIP_ERM_HPARAMS:-0}"

STAGE1_LR_GRID="${STAGE1_LR_GRID:-1e-2,1e-3,1e-4,5e-5,2e-5}"
STAGE1_WEIGHT_DECAY_GRID="${STAGE1_WEIGHT_DECAY_GRID:-1e-4,1e-5}"
STAGE1_BATCH_SIZE_GRID="${STAGE1_BATCH_SIZE_GRID:-32,64,128,256}"
STAGE2_DEMO_EMBED_DIM_GRID="${STAGE2_DEMO_EMBED_DIM_GRID:-8,16,32}"
STAGE3_LAMBDA_CV_GRID="${STAGE3_LAMBDA_CV_GRID:-0.5,1.0}"
STAGE3_ALPHA_GRID="${STAGE3_ALPHA_GRID:-25,50,75}"
STAGE4_LAMBDA_IMP_GRID="${STAGE4_LAMBDA_IMP_GRID:-0.0001,0.001,0.01,0.1, 0.3, 0.5,0.7,0.9}"
STAGE4_LAMBDA_WORST_GRID="${STAGE4_LAMBDA_WORST_GRID:-0.0001,0.001,0.01,0.1, 0.3, 0.5,0.7,0.9}"
STAGE4_LAMBDA_KD_GRID="${STAGE4_LAMBDA_KD_GRID:-0.5,0.7,0.9,1.0,1.25,1.5}"
STAGE4_TEMPERATURE_GRID="${STAGE4_TEMPERATURE_GRID:-3,4,5,6}"
STAGE4_GAMMA_GRID="${STAGE4_GAMMA_GRID:-0.5,1.0}"
STAGE4_HEAD_LR_GRID="${STAGE4_HEAD_LR_GRID:-5e-5,1e-4}"
STAGE4_ENCODER_LR_GRID="${STAGE4_ENCODER_LR_GRID:-1e-5,2e-5}"
# ResNet50 stages are 1–4 only (layer1..layer4); ViT-B/16 uses 1–12
STAGE4_UNFREEZE_FROM_BLOCK_GRID="${STAGE4_UNFREEZE_FROM_BLOCK_GRID:-2,3,4}"
CNN_STAGE4_UNFREEZE_FROM_BLOCK="${STAGE4_UNFREEZE_FROM_BLOCK}"
CNN_STAGE4_UNFREEZE_FROM_BLOCK_GRID="${STAGE4_UNFREEZE_FROM_BLOCK_GRID}"
VIT_STAGE4_UNFREEZE_FROM_BLOCK="${VIT_STAGE4_UNFREEZE_FROM_BLOCK:-8}"
VIT_STAGE4_UNFREEZE_FROM_BLOCK_GRID="${VIT_STAGE4_UNFREEZE_FROM_BLOCK_GRID:-8,9,10,11,12}"

SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
FINETUNE_ONLY="${FINETUNE_ONLY:-0}"
BASELINE_ONLY="${BASELINE_ONLY:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
mkdir -p "${TORCH_HOME}/hub/checkpoints" "${OUTPUT_DIR}"
export TORCH_HOME

_effective_run_name() {
  local base="$1" fusion="$2"
  for existing in film scalar; do
    base="${base%__fusion_${existing}}"
  done
  echo "${base}__fusion_${fusion}"
}

resolve_seed_list() {
  PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" python3 - "${SEEDS}" "${NUM_RANDOM_SEEDS}" "${SEEDS_FILE}" <<'PY'
import sys
from pathlib import Path
from chexpert_staged_fairness_pipeline import resolve_seeds
seeds_arg, num_random, seeds_file = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
seeds = resolve_seeds(
    seeds_arg,
    num_random,
    seeds_file if seeds_file.is_file() else None,
    seeds_file,
)
print(",".join(str(s) for s in seeds))
PY
}

load_finetune_summary_into_env() {
  local SUMMARY_PATH="$1"
  local tsv
  if [[ ! -f "${SUMMARY_PATH}" ]]; then
    echo "ERROR: finetune summary not found: ${SUMMARY_PATH}" >&2
    return 1
  fi
  tsv=$(python3 - "${SUMMARY_PATH}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    s = json.load(f)
s1 = s["stage1"]["hparams"]
s2 = s["stage2"]["hparams"]
s4 = s.get("stage4", s.get("stage5", {})).get("hparams", {})
s3 = s.get("stage3", {}).get("hparams", {})
fields = [
    s1["lr"], s1["weight_decay"], s1.get("batch_size", 128), s2["demo_embed_dim"],
    s4.get("lambda_cv", s3.get("lambda_cv", 0.5)),
    s4.get("alpha", s3.get("alpha", 50)),
    s4.get("lambda_imp", 0.2), s4.get("lambda_worst", 0.1),
    s4.get("lambda_kd", 0.5), s4.get("temperature", 3.0),
    s4.get("gamma", 0.5),
    s4.get("stage4_head_lr", s4.get("stage5_head_lr", 1e-4)),
    s4.get("stage4_encoder_lr", s4.get("stage5_encoder_lr", 2e-5)),
    s4.get("stage4_epochs", s4.get("stage5_epochs", 20)),
    s4.get("stage4_unfreeze_from_block", s4.get("stage5_unfreeze_from_block", 4)),
]
print("\t".join(str(x) for x in fields))
PY
)
  IFS=$'\t' read -r \
    STAGE1_LR WEIGHT_DECAY BATCH_SIZE DEMO_EMBED_DIM LAMBDA_CV ALPHA \
    LAMBDA_IMP LAMBDA_WORST LAMBDA_KD TEMPERATURE GAMMA STAGE4_HEAD_LR STAGE4_ENCODER_LR \
    STAGE4_EPOCHS STAGE4_UNFREEZE_FROM_BLOCK \
    <<< "${tsv}"
  echo "Loaded best hparams from: ${SUMMARY_PATH}"
}

build_extra_flags() {
  EXTRA_FLAGS=()
  [[ "${PRETRAINED}" == "0" ]] && EXTRA_FLAGS+=(--no-pretrained)
  [[ "${STAGE4_UNFREEZE_ENCODER}" == "1" ]] && EXTRA_FLAGS+=(--stage4-unfreeze-encoder) || EXTRA_FLAGS+=(--no-stage4-unfreeze-encoder)
  [[ "${STAGE4_UNFREEZE_INCLUDE_FC}" == "0" ]] && EXTRA_FLAGS+=(--no-stage4-unfreeze-fc)
  [[ "${USE_COSINE_SCHEDULE_STAGE4}" == "0" ]] && EXTRA_FLAGS+=(--no-cosine-schedule-stage4)
  [[ "${FAIRNESS_USE_RELU}" == "1" ]] && EXTRA_FLAGS+=(--fairness-use-relu)
  [[ "${HEAD_LAYERNORM}" == "1" ]] && EXTRA_FLAGS+=(--head-layernorm)
  [[ -n "${ERM_HPARAMS_DIR}" ]] && EXTRA_FLAGS+=(--erm-hparams-dir "${ERM_HPARAMS_DIR}")
  [[ "${SKIP_ERM_HPARAMS}" == "1" ]] && EXTRA_FLAGS+=(--skip-erm-hparams)
  EXTRA_FLAGS+=(--skip-baseline)
}

run_phase0_baselines() {
  local backbone="$1"
  echo ""
  echo "########## Phase 0: image-only ERM baselines (backbone=${backbone}) ##########"
  for seed in "${SEED_LIST[@]}"; do
    echo "  [Phase0] backbone=${backbone} seed=${seed}"
    PHASE0_ARGS=(
      --data-root "${DATA_ROOT}"
      --output-dir "${OUTPUT_DIR}"
      --checkpoint-dir "${CHECKPOINT_DIR}"
      --run-name "phase0_baseline__${backbone}"
      --backbone "${backbone}"
      --demographic-mode sex
      --num-subgroups 2
      --train-diseases "${TRAIN_DISEASES}"
      --image-size "${IMAGE_SIZE}"
      --batch-size "${BATCH_SIZE}"
      --num-workers "${NUM_WORKERS}"
      --feature-dim "${FEATURE_DIM}"
      --weight-decay "${WEIGHT_DECAY}"
      --stage1-epochs "${STAGE1_EPOCHS}"
      --stage1-lr "${STAGE1_LR}"
      --seed "${seed}"
      --device "${DEVICE}"
      --baseline-only
    )
    [[ "${PRETRAINED}" == "0" ]] && PHASE0_ARGS+=(--no-pretrained)
    [[ -n "${ERM_HPARAMS_DIR}" ]] && PHASE0_ARGS+=(--erm-hparams-dir "${ERM_HPARAMS_DIR}")
    [[ "${SKIP_ERM_HPARAMS}" == "1" ]] && PHASE0_ARGS+=(--skip-erm-hparams)
    [[ "${SKIP_BASELINE}" == "1" ]] && PHASE0_ARGS+=(--skip-baseline)
    python3 "${TRAIN_SCRIPT}" "${PHASE0_ARGS[@]}"
  done
}

common_train_args() {
  local seed="$1" run_name="$2" demo_mode="$3" num_subgroups="$4"
  build_extra_flags
  TRAIN_ARGS=(
    --data-root "${DATA_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --run-name "${run_name}"
    --backbone "${backbone}"
    --demographic-mode "${demo_mode}"
    --num-subgroups "${num_subgroups}"
    --train-diseases "${TRAIN_DISEASES}"
    --image-size "${IMAGE_SIZE}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --feature-dim "${FEATURE_DIM}"
    --demo-embed-dim "${DEMO_EMBED_DIM}"
    --gamma "${GAMMA}"
    --fusion-type "${FUSION_TYPE}"
    --residual-blocks "${RESIDUAL_BLOCKS}"
    --weight-decay "${WEIGHT_DECAY}"
    --stage1-epochs "${STAGE1_EPOCHS}"
    --stage2-epochs "${STAGE2_EPOCHS}"
    --stage4-epochs "${STAGE4_EPOCHS}"
    --stage1-lr "${STAGE1_LR}"
    --stage2-lr "${STAGE2_LR}"
    --stage4-head-lr "${STAGE4_HEAD_LR}"
    --stage4-encoder-lr "${STAGE4_ENCODER_LR}"
    --stage4-unfreeze-from-block "${STAGE4_UNFREEZE_FROM_BLOCK}"
    --warmup-epochs-stage4 "${WARMUP_EPOCHS_STAGE4}"
    --stage4-checkpoint-skip-epochs "${STAGE4_CHECKPOINT_SKIP_EPOCHS}"
    --min-lr-ratio "${MIN_LR_RATIO}"
    --alpha "${ALPHA}"
    --lambda-cv "${LAMBDA_CV}"
    --lambda-imp "${LAMBDA_IMP}"
    --lambda-worst "${LAMBDA_WORST}"
    --lambda-kd "${LAMBDA_KD}"
    --temperature "${TEMPERATURE}"
    --selection-metric-stage1 "${SELECTION_METRIC_STAGE1}"
    --selection-metric-stage2 "${SELECTION_METRIC_STAGE2}"
    --selection-metric-stage4 "${SELECTION_METRIC_STAGE4}"
    --tie-break "${TIE_BREAK}"
    --seed "${seed}"
    --device "${DEVICE}"
    "${EXTRA_FLAGS[@]}"
  )
}

echo "========== CheXpert ContextFair (shared seeds + staged finetune) =========="
echo "Output: ${OUTPUT_DIR}"
echo "Seeds file: ${SEEDS_FILE}"
echo "ERM hparams: ${ERM_HPARAMS_DIR}"
echo "Demo modes: ${DEMO_MODES}"
echo "Backbones: ${BACKBONES}"
echo "Train diseases: ${TRAIN_DISEASES}"
echo "Stage4 selection: ${SELECTION_METRIC_STAGE4} (macro AUC over 5 competition pathologies)"

RESOLVED_SEEDS="$(resolve_seed_list)"
FINETUNE_SEED="${RESOLVED_SEEDS%%,*}"
IFS=',' read -r -a SEED_LIST <<< "${RESOLVED_SEEDS}"
echo "Shared seeds (${#SEED_LIST[@]}): ${RESOLVED_SEEDS}"
echo "Finetune seed (seeds[0]): ${FINETUNE_SEED}"

IFS=',' read -r -a DEMO_LIST <<< "${DEMO_MODES}"
IFS=',' read -r -a BACKBONE_LIST <<< "${BACKBONES}"

for backbone in "${BACKBONE_LIST[@]}"; do
  backbone="$(echo "${backbone}" | xargs)"
  [[ -z "${backbone}" ]] && continue
  run_phase0_baselines "${backbone}"
done

if [[ "${BASELINE_ONLY}" == "1" ]]; then
  echo ""
  echo "BASELINE_ONLY=1; done after Phase 0 baselines."
  echo "  Seeds:   ${SEEDS_FILE}"
  echo "  Results: ${OUTPUT_DIR}/results/ERM/"
  echo "  Ckpts:   ${OUTPUT_DIR}/checkpoints/ERM/"
  exit 0
fi
for mode in "${DEMO_LIST[@]}"; do
  mode="$(echo "${mode}" | xargs)"
  [[ -z "${mode}" ]] && continue
  case "${mode}" in
    sex)
      demo_mode="sex"
      num_subgroups=2
      ;;
    age)
      demo_mode="age"
      num_subgroups=2
      ;;
    *)
      echo "ERROR: unknown DEMO_MODE='${mode}' (use sex|age)" >&2
      exit 1
      ;;
  esac

  for backbone in "${BACKBONE_LIST[@]}"; do
    backbone="$(echo "${backbone}" | xargs)"
    [[ -z "${backbone}" ]] && continue
    case "${backbone}" in
      resnet50)
        STAGE4_UNFREEZE_FROM_BLOCK="${CNN_STAGE4_UNFREEZE_FROM_BLOCK}"
        STAGE4_UNFREEZE_FROM_BLOCK_GRID="${CNN_STAGE4_UNFREEZE_FROM_BLOCK_GRID}"
        ;;
      vit_b_16)
        STAGE4_UNFREEZE_FROM_BLOCK="${VIT_STAGE4_UNFREEZE_FROM_BLOCK}"
        STAGE4_UNFREEZE_FROM_BLOCK_GRID="${VIT_STAGE4_UNFREEZE_FROM_BLOCK_GRID}"
        ;;
      *)
        echo "ERROR: unknown BACKBONE='${backbone}' (use resnet50|vit_b_16)" >&2
        exit 1
        ;;
    esac
    run_name="${demo_mode}_proposed__${backbone}"
    effective_run="$(_effective_run_name "${run_name}" "${FUSION_TYPE}")"
    summary_path="${OUTPUT_DIR}/results/${effective_run}/${demo_mode}/finetune/finetune_summary.json"

    echo ""
    echo "######################################################################"
    echo "# ContextFair | demographic=${demo_mode} | backbone=${backbone}"
    echo "# Stage4 unfreeze grid: ${STAGE4_UNFREEZE_FROM_BLOCK_GRID}"
    echo "######################################################################"

    if [[ "${SKIP_FINETUNE}" != "1" ]]; then
      echo ""
      echo "########## Phase 1: finetune on seed ${FINETUNE_SEED} ##########"
      build_extra_flags
      python3 "${TRAIN_SCRIPT}" \
        --finetune-stages \
        --finetune-sequential-lambdas \
        --finetune-only-stage all \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUTPUT_DIR}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --run-name "${run_name}" \
        --backbone "${backbone}" \
        --demographic-mode "${demo_mode}" \
        --num-subgroups "${num_subgroups}" \
        --train-diseases "${TRAIN_DISEASES}" \
        --image-size "${IMAGE_SIZE}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --feature-dim "${FEATURE_DIM}" \
        --demo-embed-dim "${DEMO_EMBED_DIM}" \
        --gamma "${GAMMA}" \
        --fusion-type "${FUSION_TYPE}" \
        --residual-blocks "${RESIDUAL_BLOCKS}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --stage1-epochs "${STAGE1_EPOCHS}" \
        --stage2-epochs "${STAGE2_EPOCHS}" \
        --stage4-epochs "${STAGE4_EPOCHS}" \
        --stage1-lr "${STAGE1_LR}" \
        --stage2-lr "${STAGE2_LR}" \
        --stage4-head-lr "${STAGE4_HEAD_LR}" \
        --stage4-encoder-lr "${STAGE4_ENCODER_LR}" \
        --stage4-unfreeze-from-block "${STAGE4_UNFREEZE_FROM_BLOCK}" \
        --warmup-epochs-stage4 "${WARMUP_EPOCHS_STAGE4}" \
        --stage4-checkpoint-skip-epochs "${STAGE4_CHECKPOINT_SKIP_EPOCHS}" \
        --min-lr-ratio "${MIN_LR_RATIO}" \
        --alpha "${ALPHA}" \
        --lambda-cv "${LAMBDA_CV}" \
        --selection-metric-stage1 "${SELECTION_METRIC_STAGE1}" \
        --selection-metric-stage2 "${SELECTION_METRIC_STAGE2}" \
        --selection-metric-stage4 "${SELECTION_METRIC_STAGE4}" \
        --tie-break "${TIE_BREAK}" \
        --stage1-lr-grid "${STAGE1_LR_GRID}" \
        --stage1-weight-decay-grid "${STAGE1_WEIGHT_DECAY_GRID}" \
        --stage1-batch-size-grid "${STAGE1_BATCH_SIZE_GRID}" \
        --stage2-demo-embed-dim-grid "${STAGE2_DEMO_EMBED_DIM_GRID}" \
        --stage3-lambda-cv-grid "${STAGE3_LAMBDA_CV_GRID}" \
        --stage3-alpha-grid "${STAGE3_ALPHA_GRID}" \
        --stage4-lambda-imp-grid "${STAGE4_LAMBDA_IMP_GRID}" \
        --stage4-lambda-worst-grid "${STAGE4_LAMBDA_WORST_GRID}" \
        --stage4-lambda-kd-grid "${STAGE4_LAMBDA_KD_GRID}" \
        --stage4-temperature-grid "${STAGE4_TEMPERATURE_GRID}" \
        --stage4-gamma-grid "${STAGE4_GAMMA_GRID}" \
        --stage4-head-lr-grid "${STAGE4_HEAD_LR_GRID}" \
        --stage4-encoder-lr-grid "${STAGE4_ENCODER_LR_GRID}" \
        --stage4-unfreeze-from-block-grid "${STAGE4_UNFREEZE_FROM_BLOCK_GRID}" \
        --seed "${FINETUNE_SEED}" \
        --device "${DEVICE}" \
        "${EXTRA_FLAGS[@]}"
      echo "Finetune summary: ${summary_path}"
    else
      echo "SKIP_FINETUNE=1; reusing existing summary"
    fi

    if [[ "${FINETUNE_ONLY}" == "1" ]]; then
      echo "FINETUNE_ONLY=1; skipping multi-seed training for ${demo_mode}/${backbone}."
      continue
    fi

    load_finetune_summary_into_env "${summary_path}"

    echo ""
    echo "########## Phase 2: train all seeds with best hparams ##########"
    for seed in "${SEED_LIST[@]}"; do
      echo ""
      echo "========== ContextFair | ${demo_mode} | ${backbone} | seed=${seed} =========="
      common_train_args "${seed}" "${run_name}" "${demo_mode}" "${num_subgroups}"
      python3 "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
    done
  done
done

echo ""
echo "Done."
echo "  Seeds:   ${SEEDS_FILE}"
echo "  Results: ${OUTPUT_DIR}/results/"
echo "  Ckpts:   ${CHECKPOINT_DIR}/"
