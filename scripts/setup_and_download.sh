#!/bin/bash

# Run once on the MANIT login node. Creates the Python environment, caches
# pretrained weights, and downloads raw data into persistent storage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$HOME/AstraGuard-LandClassifier-data}"
RAW_DIR="${RAW_DIR:-${DATA_ROOT}/raw}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/astraguard-landcover}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/astraguard-landcover/bin/python}"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
BOOTSTRAP_MARKER="${CACHE_DIR}/setup_complete_v2"
AOI_MANIFEST="${AOI_MANIFEST:-configs/mp_aois.tsv}"
MAX_RAW_GB="${MAX_RAW_GB:-50}"
SENTINEL_MAX_SCENES="${SENTINEL_MAX_SCENES:-12}"
SENTINEL_MAX_CLOUD="${SENTINEL_MAX_CLOUD:-20}"

if [[ "${AOI_MANIFEST}" != /* ]]; then
  AOI_MANIFEST="${PROJECT_ROOT}/${AOI_MANIFEST}"
fi

mkdir -p "${RAW_DIR}" "${CACHE_DIR}"
cd "${PROJECT_ROOT}"

echo "AstraGuard setup and raw-data download"
echo "  project: ${PROJECT_ROOT}"
echo "  raw:     ${RAW_DIR}"
echo "  cache:   ${CACHE_DIR}"
echo "  python:  ${PYTHON_BIN}"
echo "  AOIs:    ${AOI_MANIFEST}"
echo "  limit:   ${MAX_RAW_GB} GiB raw data"

environment_ready() {
  [[ -x "${PYTHON_BIN}" && -f "${BOOTSTRAP_MARKER}" ]] \
    && "${PYTHON_BIN}" -c \
    'from importlib.metadata import version
import astraguard_landcover, h5py, planetary_computer, rasterio, rioxarray, stackstac
assert version("transformers") == "4.48.3"
assert version("torch") == "2.6.0+cu118"
assert version("torchvision") == "0.21.0+cu118"' \
    >/dev/null 2>&1
}

if ! environment_ready; then
  if [[ "${BOOTSTRAP_ENV}" != "1" ]]; then
    echo "Python environment is missing or incomplete and BOOTSTRAP_ENV=0: ${PYTHON_BIN}" >&2
    exit 2
  fi
  echo "Creating or completing the Conda environment and caching model weights..."
  ENV_NAME=astraguard-landcover \
  CACHE_DIR="${CACHE_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}" \
  PREFETCH_MODELS=1 \
  bash scripts/bootstrap_hpc.sh
fi

if ! environment_ready; then
  echo "Python environment is still incomplete at ${PYTHON_BIN}" >&2
  exit 3
fi

"${PYTHON_BIN}" -m astraguard_landcover.download_manifest \
  --manifest "${AOI_MANIFEST}" \
  --output-root "${RAW_DIR}" \
  --max-raw-gb "${MAX_RAW_GB}" \
  --max-scenes "${SENTINEL_MAX_SCENES}" \
  --max-cloud "${SENTINEL_MAX_CLOUD}"

echo
echo "Environment and raw data are ready."
echo "Next command: bash scripts/submit_pipeline.sh"
