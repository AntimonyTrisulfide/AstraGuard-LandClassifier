#!/bin/bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-astraguard-landcover}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/astraguard-landcover}"
PREFETCH_MODELS="${PREFETCH_MODELS:-1}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/${ENV_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if [[ -z "${CONDA_EXE}" || ! -x "${CONDA_EXE}" ]]; then
    echo "Conda executable not found. Set CONDA_EXE to its absolute path." >&2
    exit 2
  fi
  "${CONDA_EXE}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
  echo "Reusing existing Conda environment: ${ENV_PREFIX}"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
# MANIT's older compute-node runtime cannot use the newest h5py/NumPy wheels.
# Pin compatible manylinux2014 binaries so pip never falls back to a local
# source build (the cluster image does not provide a C++ compiler).
"${PYTHON_BIN}" -m pip install --only-binary=:all: \
  "numpy==1.26.4" \
  "h5py==3.10.0"
"${PYTHON_BIN}" -m pip install -e ".[models,geo,dev]"
"${PYTHON_BIN}" -m unittest discover -s tests -v

mkdir -p "${CACHE_DIR}"
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
if [[ "${PREFETCH_MODELS}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/prefetch_models.py \
    configs/deeplabv3plus.yaml configs/segformer.yaml
fi

echo "Environment ${ENV_NAME} is ready."
echo "Set PYTHON_BIN=${PYTHON_BIN} before submitting PBS jobs."
echo "Set CACHE_DIR=${CACHE_DIR} before submitting PBS jobs."
