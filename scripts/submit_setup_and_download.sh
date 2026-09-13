#!/bin/bash

# Submit environment setup and raw-data acquisition to a CPU PBS queue.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$HOME/AstraGuard-LandClassifier-data}"
RAW_DIR="${RAW_DIR:-${DATA_ROOT}/raw}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/astraguard-landcover}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/astraguard-landcover/bin/python}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
DOWNLOAD_QUEUE="${DOWNLOAD_QUEUE:-long}"
DOWNLOAD_WALLTIME="${DOWNLOAD_WALLTIME:-24:00:00}"

if [[ -z "${CONDA_EXE}" || ! -x "${CONDA_EXE}" ]]; then
  echo "Conda executable not found. Set CONDA_EXE to its absolute path." >&2
  exit 2
fi

mkdir -p "${DATA_ROOT}/logs" "${RAW_DIR}" "${CACHE_DIR}"
cd "${PROJECT_ROOT}"

qsub \
  -q "${DOWNLOAD_QUEUE}" \
  -l "walltime=${DOWNLOAD_WALLTIME}" \
  -v \
DATA_ROOT="${DATA_ROOT}",\
RAW_DIR="${RAW_DIR}",\
CACHE_DIR="${CACHE_DIR}",\
PYTHON_BIN="${PYTHON_BIN}",\
CONDA_EXE="${CONDA_EXE}" \
"${PROJECT_ROOT}/scripts/setup_and_download.pbs"
