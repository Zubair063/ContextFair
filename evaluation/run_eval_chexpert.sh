#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRED="${PRED:?Set PRED=path/to/results/ERM_...}"
BASE="${BASE:-}"
DEMOGRAPHIC="${DEMOGRAPHIC:-sex}"
OUT="${OUT:-${PRED}/metrics}"

CMD=(
  python3 "${SCRIPT_DIR}/compute_metrics.py"
  --predictions-dir "${PRED}"
  --demographic "${DEMOGRAPHIC}"
  --output-dir "${OUT}"
)
if [[ -n "${BASE}" ]]; then
  CMD+=(--baseline-dir "${BASE}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
echo "Metrics written under: ${OUT}"
