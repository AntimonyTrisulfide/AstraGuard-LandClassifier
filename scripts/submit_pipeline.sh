#!/bin/bash

# Submit the single preprocessing + training + evaluation PBS allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$HOME/AstraGuard-LandClassifier-data}"
RAW_DIR="${RAW_DIR:-${DATA_ROOT}/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/runs}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/astraguard-landcover}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/astraguard-landcover/bin/python}"
CONFIG="${CONFIG:-configs/deeplabv3plus.yaml}"
RUN_NAME="${RUN_NAME:-deeplabv3plus_v1}"
RESUME="${RESUME:-${OUTPUT_DIR}/${RUN_NAME}/last.pt}"
TILE_SIZE="${TILE_SIZE:-256}"
STRIDE="${STRIDE:-256}"
MIN_VALID_FRACTION="${MIN_VALID_FRACTION:-0.95}"
GPU_QUEUE="${GPU_QUEUE:-dgx}"
GPU_HOST="${GPU_HOST:-gpu2}"
GPU_NCPUS="${GPU_NCPUS:-12}"
GPU_MEM="${GPU_MEM:-24gb}"
GPU_WALLTIME="${GPU_WALLTIME:-24:00:00}"
GPU_AFTER_JOB_ID="${GPU_AFTER_JOB_ID:-}"
AOI_MANIFEST="${AOI_MANIFEST:-configs/mp_aois.tsv}"

if [[ "${AOI_MANIFEST}" != /* ]]; then
  AOI_MANIFEST="${PROJECT_ROOT}/${AOI_MANIFEST}"
fi

case "${GPU_QUEUE}" in
  dgx|max_dgx) ;;
  *)
    echo "Invalid GPU_QUEUE '${GPU_QUEUE}'. Choose dgx or max_dgx." >&2
    exit 2
    ;;
esac
if [[ -n "${GPU_HOST}" && ! "${GPU_HOST}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid GPU_HOST '${GPU_HOST}'." >&2
  exit 2
fi
if [[ ! "${GPU_NCPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU_NCPUS must be a positive integer." >&2
  exit 2
fi
if [[ -n "${GPU_AFTER_JOB_ID}" && ! "${GPU_AFTER_JOB_ID}" =~ ^[0-9]+([.][A-Za-z0-9._-]+)?$ ]]; then
  echo "Invalid GPU_AFTER_JOB_ID '${GPU_AFTER_JOB_ID}'." >&2
  exit 2
fi

SELECT_RESOURCE="select=1:ncpus=${GPU_NCPUS}:mem=${GPU_MEM}:ngpus=1"
if [[ -n "${GPU_HOST}" ]]; then
  SELECT_RESOURCE+=":host=${GPU_HOST}"
fi
QSUB_DEPENDENCY=()
if [[ -n "${GPU_AFTER_JOB_ID}" ]]; then
  if ! qstat "${GPU_AFTER_JOB_ID}" >/dev/null 2>&1; then
    echo "Dependency job is not active: ${GPU_AFTER_JOB_ID}" >&2
    exit 2
  fi
  QSUB_DEPENDENCY=(-W "depend=afterany:${GPU_AFTER_JOB_ID}")
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment is missing: ${PYTHON_BIN}" >&2
  echo "Run first: bash scripts/setup_and_download.sh" >&2
  exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/${CONFIG}" && ! -f "${CONFIG}" ]]; then
  echo "Training config does not exist: ${CONFIG}" >&2
  exit 3
fi
if ! "${PYTHON_BIN}" -m astraguard_landcover.download_manifest \
  --manifest "${AOI_MANIFEST}" \
  --output-root "${RAW_DIR}" \
  --check-only; then
  echo "Raw data is incomplete. Run: bash scripts/submit_setup_and_download.sh" >&2
  exit 4
fi

mkdir -p "${OUTPUT_DIR}/${RUN_NAME}/logs" "${CACHE_DIR}"
cd "${PROJECT_ROOT}"

qsub -q "${GPU_QUEUE}" \
  -l "${SELECT_RESOURCE}" \
  -l "walltime=${GPU_WALLTIME}" \
  "${QSUB_DEPENDENCY[@]}" \
  -v \
PYTHON_BIN="${PYTHON_BIN}",\
DATA_ROOT="${DATA_ROOT}",\
RAW_DIR="${RAW_DIR}",\
OUTPUT_DIR="${OUTPUT_DIR}",\
CACHE_DIR="${CACHE_DIR}",\
CONFIG="${CONFIG}",\
RUN_NAME="${RUN_NAME}",\
RESUME="${RESUME}",\
TILE_SIZE="${TILE_SIZE}",\
STRIDE="${STRIDE}",\
MIN_VALID_FRACTION="${MIN_VALID_FRACTION}",\
AOI_MANIFEST="${AOI_MANIFEST}" \
"${PROJECT_ROOT}/scripts/pipeline.pbs"
