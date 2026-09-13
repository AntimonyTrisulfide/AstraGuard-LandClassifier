#!/bin/bash

# Submit the single preprocessing + training + evaluation PBS allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$HOME/AstraGuard-LandClassifier-data}"
RAW_DIR="${RAW_DIR:-${DATA_ROOT}/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/runs}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/astraguard-landcover}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/astraguard-landcover/bin/python}"
CONFIG="${CONFIG:-configs/deeplabv3plus.yaml}"
RUN_NAME="${RUN_NAME:-deeplabv3plus_v1}"
RESUME="${RESUME:-${OUTPUT_DIR}/${RUN_NAME}/last.pt}"
TILE_SIZE="${TILE_SIZE:-256}"
STRIDE="${STRIDE:-256}"
MIN_VALID_FRACTION="${MIN_VALID_FRACTION:-0.95}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment is missing: ${PYTHON_BIN}" >&2
  echo "Run first: bash scripts/setup_and_download.sh" >&2
  exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/${CONFIG}" && ! -f "${CONFIG}" ]]; then
  echo "Training config does not exist: ${CONFIG}" >&2
  exit 3
fi
for region_id in bhopal_train indore_val jabalpur_test; do
  for required in image.tif worldcover.tif metadata.json; do
    if [[ ! -f "${RAW_DIR}/${region_id}/${required}" ]]; then
      echo "Raw data is missing: ${RAW_DIR}/${region_id}/${required}" >&2
      echo "Run first: bash scripts/setup_and_download.sh" >&2
      exit 4
    fi
  done
done

mkdir -p "${OUTPUT_DIR}/${RUN_NAME}/logs" "${CACHE_DIR}"
cd "${PROJECT_ROOT}"

qsub -v \
PYTHON_BIN="${PYTHON_BIN}",\
DATA_ROOT="${DATA_ROOT}",\
RAW_DIR="${RAW_DIR}",\
OUTPUT_DIR="${OUTPUT_DIR}",\
CACHE_DIR="${CACHE_DIR}",\
CONFIG="${CONFIG}",\
RUN_NAME="${RUN_NAME}",\
RESUME="${RESUME}",\
TILE_SIZE="${TILE_SIZE}",\
STRIDE="${STRIDE}",\
MIN_VALID_FRACTION="${MIN_VALID_FRACTION}" \
"${PROJECT_ROOT}/scripts/pipeline.pbs"

