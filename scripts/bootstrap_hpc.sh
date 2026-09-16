#!/bin/bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-astraguard-landcover}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/astraguard-landcover}"
PREFETCH_MODELS="${PREFETCH_MODELS:-1}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/${ENV_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
BOOTSTRAP_MARKER="${CACHE_DIR}/setup_complete_v2"

mkdir -p "${CACHE_DIR}"
rm -f "${BOOTSTRAP_MARKER}"

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
# MANIT's older compute-node runtime cannot use the newest scientific and
# geospatial wheels. Pin compatible manylinux2014 binaries so pip never falls
# back to a local source build (the cluster image has no C++/GDAL toolchain).
"${PYTHON_BIN}" -m pip install --only-binary=:all: \
  "numpy==1.26.4" \
  "h5py==3.10.0" \
  "rasterio==1.3.10" \
  "contourpy==1.2.1"
# The MANIT dgx node currently exposes NVIDIA driver 460.32.03. PyTorch's
# CUDA 12.4 wheel cannot initialize on that driver, so use the CUDA 11.8
# build explicitly. Exact local-version pins prevent an installed +cu124
# wheel from satisfying the requirement accidentally.
if ! "${PYTHON_BIN}" -c \
  'import torch, torchvision
assert torch.__version__ == "2.6.0+cu118"
assert torchvision.__version__ == "0.21.0+cu118"' \
  >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --only-binary=:all: --upgrade --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.6.0+cu118" \
    "torchvision==0.21.0+cu118"
fi
# Compute nodes do not provide a build toolchain. Restrict every third-party
# dependency to wheels so pip can select an older compatible binary instead of
# repeatedly attempting source builds.
"${PYTHON_BIN}" -m pip install --only-binary=:all: -e ".[models,geo,dev]"
"${PYTHON_BIN}" -m unittest discover -s tests -v

export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
if [[ "${PREFETCH_MODELS}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/prefetch_models.py \
    configs/deeplabv3plus.yaml configs/segformer.yaml
  touch "${BOOTSTRAP_MARKER}"
fi

echo "Environment ${ENV_NAME} is ready."
echo "Set PYTHON_BIN=${PYTHON_BIN} before submitting PBS jobs."
echo "Set CACHE_DIR=${CACHE_DIR} before submitting PBS jobs."
