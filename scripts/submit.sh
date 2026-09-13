#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/astraguard-landcover/bin/python}"
CONFIG="${CONFIG:-configs/deeplabv3plus.yaml}"
RUN_NAME="${RUN_NAME:-deeplabv3plus_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs}"
PROCESSED_DIR="${PROCESSED_DIR:?Set PROCESSED_DIR to the persistent HDF5 directory}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/astraguard-landcover}"
STAGE_DATA="${STAGE_DATA:-1}"
RESUME="${RESUME:-${OUTPUT_DIR}/${RUN_NAME}/last.pt}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/${CONFIG}" && ! -f "${CONFIG}" ]]; then
  echo "Training config does not exist: ${CONFIG}" >&2
  exit 3
fi
for required in train.h5 val.h5 stats.json manifest.json; do
  if [[ ! -f "${PROCESSED_DIR}/${required}" ]]; then
    echo "Required dataset file does not exist: ${PROCESSED_DIR}/${required}" >&2
    exit 4
  fi
done

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"
qsub -v \
PYTHON_BIN="${PYTHON_BIN}",\
CONFIG="${CONFIG}",\
RUN_NAME="${RUN_NAME}",\
OUTPUT_DIR="${OUTPUT_DIR}",\
PROCESSED_DIR="${PROCESSED_DIR}",\
CACHE_DIR="${CACHE_DIR}",\
STAGE_DATA="${STAGE_DATA}",\
RESUME="${RESUME}" \
"${PROJECT_ROOT}/scripts/train.pbs"

