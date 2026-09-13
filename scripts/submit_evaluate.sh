#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/astraguard-landcover/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the best.pt file}"
PROCESSED_DIR="${PROCESSED_DIR:?Set PROCESSED_DIR to the persistent HDF5 directory}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$(dirname "${CHECKPOINT}")/test}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/astraguard-landcover}"
STAGE_DATA="${STAGE_DATA:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi
for required in "${CHECKPOINT}" "${PROCESSED_DIR}/test.h5" \
  "${PROCESSED_DIR}/stats.json" "${PROCESSED_DIR}/manifest.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file does not exist: ${required}" >&2
    exit 3
  fi
done

mkdir -p "${EVAL_OUTPUT_DIR}" "${CACHE_DIR}" "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"
qsub -v \
PYTHON_BIN="${PYTHON_BIN}",\
CHECKPOINT="${CHECKPOINT}",\
PROCESSED_DIR="${PROCESSED_DIR}",\
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}",\
CACHE_DIR="${CACHE_DIR}",\
STAGE_DATA="${STAGE_DATA}" \
"${PROJECT_ROOT}/scripts/evaluate.pbs"

