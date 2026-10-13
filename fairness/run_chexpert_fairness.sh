#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERM_DIR="$(cd "${SCRIPT_DIR}/../ERM" && pwd)"

GPU_ID="${GPU_ID:-}"
METHODS="${METHODS:-fairdro,fairread,fairsde,fairtune,groupdro,laftr}"
DEMO_MODES="${DEMO_MODES:-sex,age}"
SEEDS="${SEEDS:-random}"
NUM_RANDOM_SEEDS="${NUM_RANDOM_SEEDS:-5}"

SHARED_SEEDS_FILE="${SHARED_SEEDS_FILE:-}"
ERM_HPARAMS_DIR="${ERM_HPARAMS_DIR:-${ERM_DIR}/chexpert_outputs/finetune}"
SELECTION_METRIC="${SELECTION_METRIC:-overall_macro_auc}"

LR_GRID="${LR_GRID:-1e-2,1e-3,1e-4,5e-5,2e-5}"
WD_GRID="${WD_GRID:-1e-4,1e-5}"
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-32,64,128,256}"

SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
FINETUNE_ONLY="${FINETUNE_ONLY:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
BASELINE_ONLY="${BASELINE_ONLY:-0}"

export DEMO_MODES SEEDS NUM_RANDOM_SEEDS
export ERM_HPARAMS_DIR LR_GRID WD_GRID BATCH_SIZE_GRID
export SKIP_FINETUNE FINETUNE_ONLY SKIP_BASELINE BASELINE_ONLY
export SELECTION_METRIC
if [[ -n "${GPU_ID}" ]]; then
  export GPU_ID
fi

echo "========== CheXpert Fairness suite (Fusion-style; no FairMedFM) =========="
echo "Methods: ${METHODS}"
echo "Demo modes: ${DEMO_MODES}"
echo "Seeds: ${SEEDS} (n=${NUM_RANDOM_SEEDS})"
echo "ERM hparams: ${ERM_HPARAMS_DIR}"
echo "Selection metric: ${SELECTION_METRIC}"
echo "Grids: lr=${LR_GRID} wd=${WD_GRID} bs=${BATCH_SIZE_GRID}"

IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
for method in "${METHOD_LIST[@]}"; do
  method="$(echo "${method}" | xargs | tr '[:upper:]' '[:lower:]')"
  [[ -z "${method}" ]] && continue
  case "${method}" in
    fairdro) script="run_chexpert_fairdro.sh" ;;
    fairread) script="run_chexpert_fairread.sh" ;;
    fairsde) script="run_chexpert_fairsde.sh" ;;
    fairtune) script="run_chexpert_fairtune.sh" ;;
    groupdro) script="run_chexpert_groupdro.sh" ;;
    laftr) script="run_chexpert_laftr.sh" ;;
    fairmedfm)
      echo "Skipping FairMedFM (excluded from this suite)."
      continue
      ;;
    *)
      echo "ERROR: unknown METHOD='${method}'" >&2
      echo "Valid: fairdro,fairread,fairsde,fairtune,groupdro,laftr" >&2
      exit 1
      ;;
  esac
  launcher="${SCRIPT_DIR}/${script}"
  if [[ ! -f "${launcher}" ]]; then
    echo "ERROR: missing launcher ${launcher}" >&2
    exit 1
  fi
  echo ""
  echo "######################################################################"
  echo "# Method: ${method}"
  echo "######################################################################"
  if [[ -n "${SHARED_SEEDS_FILE}" ]]; then
    SEEDS_FILE="${SHARED_SEEDS_FILE}" bash "${launcher}"
  else
    bash "${launcher}"
  fi
done

echo ""
echo "All requested fairness methods finished."
