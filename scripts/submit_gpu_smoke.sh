#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ASTRAGUARD_VENV:-$HOME/.conda/envs/astraguard-landcover}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
qsub -v PYTHON_BIN="${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/gpu_smoke.pbs"

