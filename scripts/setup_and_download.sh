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

mkdir -p "${RAW_DIR}" "${CACHE_DIR}"
cd "${PROJECT_ROOT}"

echo "AstraGuard setup and raw-data download"
echo "  project: ${PROJECT_ROOT}"
echo "  raw:     ${RAW_DIR}"
echo "  cache:   ${CACHE_DIR}"
echo "  python:  ${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if [[ "${BOOTSTRAP_ENV}" != "1" ]]; then
    echo "Python environment is missing and BOOTSTRAP_ENV=0: ${PYTHON_BIN}" >&2
    exit 2
  fi
  echo "Creating the Conda environment and caching model weights..."
  ENV_NAME=astraguard-landcover \
  CACHE_DIR="${CACHE_DIR}" \
  PREFETCH_MODELS=1 \
  bash scripts/bootstrap_hpc.sh
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment was not created at ${PYTHON_BIN}" >&2
  exit 3
fi

download_region() {
  local region_id="$1"
  local split="$2"
  local west="$3"
  local south="$4"
  local east="$5"
  local north="$6"
  local region_dir="${RAW_DIR}/${region_id}"

  if [[ -f "${region_dir}/image.tif" \
    && -f "${region_dir}/worldcover.tif" \
    && -f "${region_dir}/metadata.json" ]]; then
    echo "Data already complete; skipping ${region_id}"
    return
  fi

  echo "Downloading ${region_id}..."
  "${PYTHON_BIN}" -m astraguard_landcover.download_aoi \
    --region-id "${region_id}" \
    --split "${split}" \
    --bbox "${west}" "${south}" "${east}" "${north}" \
    --output-root "${RAW_DIR}"
}

download_region bhopal_train train 77.30 23.15 77.50 23.35
download_region indore_val val 75.75 22.62 75.95 22.82
download_region jabalpur_test test 79.85 23.08 80.05 23.28

for region_id in bhopal_train indore_val jabalpur_test; do
  for required in image.tif worldcover.tif metadata.json; do
    if [[ ! -f "${RAW_DIR}/${region_id}/${required}" ]]; then
      echo "Download incomplete: ${RAW_DIR}/${region_id}/${required}" >&2
      exit 4
    fi
  done
done

echo
echo "Environment and raw data are ready."
echo "Next command: bash scripts/submit_pipeline.sh"

