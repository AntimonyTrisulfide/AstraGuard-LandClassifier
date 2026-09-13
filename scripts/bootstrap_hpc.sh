#!/bin/bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-astraguard-landcover}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/astraguard-landcover}"
PREFETCH_MODELS="${PREFETCH_MODELS:-1}"

conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -e ".[models,geo,dev]"
python -m unittest discover -s tests -v

mkdir -p "${CACHE_DIR}"
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
if [[ "${PREFETCH_MODELS}" == "1" ]]; then
  python scripts/prefetch_models.py \
    configs/deeplabv3plus.yaml configs/segformer.yaml
fi

echo "Environment ${ENV_NAME} is ready."
echo "Set PYTHON_BIN=$(which python) before submitting PBS jobs."
echo "Set CACHE_DIR=${CACHE_DIR} before submitting PBS jobs."

