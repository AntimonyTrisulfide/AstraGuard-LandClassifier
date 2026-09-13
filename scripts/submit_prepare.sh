#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/astraguard-landcover/bin/python}"
RAW_DIR="${RAW_DIR:?Set RAW_DIR to the persistent raw-region directory}"
PROCESSED_DIR="${PROCESSED_DIR:?Set PROCESSED_DIR to the persistent HDF5 directory}"
TILE_SIZE="${TILE_SIZE:-256}"
STRIDE="${STRIDE:-256}"
MIN_VALID_FRACTION="${MIN_VALID_FRACTION:-0.95}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${RAW_DIR}" ]]; then
  echo "Raw data directory does not exist: ${RAW_DIR}" >&2
  exit 3
fi

mkdir -p "${PROCESSED_DIR}" "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"
qsub -v \
PYTHON_BIN="${PYTHON_BIN}",\
RAW_DIR="${RAW_DIR}",\
PROCESSED_DIR="${PROCESSED_DIR}",\
TILE_SIZE="${TILE_SIZE}",\
STRIDE="${STRIDE}",\
MIN_VALID_FRACTION="${MIN_VALID_FRACTION}" \
"${PROJECT_ROOT}/scripts/prepare.pbs"

