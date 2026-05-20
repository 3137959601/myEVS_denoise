#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
EDFORMER_ROOT="${EDFORMER_ROOT:-/workspace/EDformer}"
BASE_PATH="${BASE_PATH:-/workspace/dataset/DND21/mydriving_ED24}"
HZ="${HZ:-1,3,5,7,10}"
DEVICE="${DEVICE:-cuda:0}"
XY_MODE="${XY_MODE:-official}"
MAX_EVENTS="${MAX_EVENTS:-0}"

OUT_DIR="data/DND21/edformer_official_auc"
mkdir -p "${OUT_DIR}"

args=(
  "scripts/EDformer_official/eval_official_driving_auc.py"
  "--edformer-root" "${EDFORMER_ROOT}"
  "--base-path" "${BASE_PATH}"
  "--hz" "${HZ}"
  "--filename" "driving_mix_result.txt"
  "--device" "${DEVICE}"
  "--xy-mode" "${XY_MODE}"
  "--skip-missing"
  "--out-csv" "${OUT_DIR}/driving_auc_${XY_MODE}.csv"
  "--out-json" "${OUT_DIR}/driving_auc_${XY_MODE}_env.json"
)

if [[ "${MAX_EVENTS}" != "0" ]]; then
  args+=("--max-events" "${MAX_EVENTS}")
fi

"${PYTHON_BIN}" "${args[@]}"
