# AstraGuard — Land Cover Segmentation

Member 1's standalone model project for multispectral Sentinel-2 land-cover
segmentation and agricultural-land mapping.

## Final MANIT workflow

There are exactly two user-facing stages.

```text
LOGIN NODE — persistent storage
Environment setup + pretrained weights + raw-data download
                         |
                         v
GPU PBS JOB (dgx or max_dgx) — job-local $TMPDIR
Preprocessing -> training -> held-out evaluation
                         |
                         v
PERSISTENT STORAGE
Checkpoints + metrics + manifests + logs
```

### First command: queue environment setup and raw-data download

Run once from the HPC login node. The work itself runs on the CPU-only
`long` queue:

```bash
cd "$HOME/AstraGuard-LandClassifier"
bash scripts/submit_setup_and_download.sh
qstat -u "$USER"
```

The submitted CPU job:

- creates the `astraguard-landcover` Conda environment when missing;
- installs the project and runs its tests;
- caches DeepLabV3+ and SegFormer pretrained assets persistently;
- downloads the three pilot Sentinel-2/WorldCover regions directly on HPC;
- skips any region that is already complete.

Its persistent log is written under:

```text
$HOME/AstraGuard-LandClassifier-data/logs/setup_download_<job-id>.log
```

Wait for this job to finish successfully before entering the second command.
If a previous attempt stopped while compiling NumPy, h5py, or Rasterio, pull
the latest version and submit this same command again. The bootstrap repairs
the existing environment using compatible binary wheels and completed
downloads are skipped.

Default persistent locations:

```text
$HOME/AstraGuard-LandClassifier-data/raw
$HOME/.cache/astraguard-landcover
```

### Second command: preprocess, train, and evaluate

This command can be entered even when no GPU is currently free:

```bash
cd "$HOME/AstraGuard-LandClassifier"
# Default GPU queue: dgx
bash scripts/submit_pipeline.sh

# Or select max_dgx
GPU_QUEUE=max_dgx bash scripts/submit_pipeline.sh
```

Submit only one of the two commands. PBS returns a job ID and leaves it queued
until a GPU is available in the selected queue. The launcher rejects every
queue except `dgx` and `max_dgx`. Inside that one allocation,
[scripts/pipeline.pbs](scripts/pipeline.pbs) performs:

1. Generate `train.h5`, `val.h5`, and `test.h5` under `$TMPDIR`.
2. Train DeepLabV3+ against those exact temporary paths.
3. Evaluate `best.pt` against the temporary held-out `test.h5`.
4. Keep checkpoints, metrics, manifests, and logs in persistent storage.

The HDF5 files disappear with PBS scratch after the job. Raw GeoTIFFs never go
into scratch permanently and can regenerate the processed dataset on any rerun.

Monitor the job:

```bash
qstat -u "$USER"
qstat -f <job-id> | egrep 'job_state|comment|resources_used.walltime|Resource_List'
```

Persistent results:

```text
$HOME/AstraGuard-LandClassifier-data/runs/deeplabv3plus_v1/
├── best.pt
├── last.pt
├── history.csv
├── dataset_manifest.json
├── dataset_stats.json
├── logs/
└── test/
    ├── metrics.json
    └── confusion_matrix.png
```

The pipeline automatically resumes `last.pt`. If PBS walltime expires, submit
the same second command again; preprocessing is regenerated in the new
`$TMPDIR`, training resumes, and evaluation follows after training completes.

## Custom persistent storage

If your allocated data location is not under `$HOME`, pass the same `DATA_ROOT`
to both stages:

```bash
DATA_ROOT=/your/allocated/path/astraguard \
bash scripts/submit_setup_and_download.sh

DATA_ROOT=/your/allocated/path/astraguard \
bash scripts/submit_pipeline.sh
```

Other optional overrides:

```bash
DATA_ROOT=/your/allocated/path/astraguard \
CONFIG=configs/deeplabv3plus.yaml \
RUN_NAME=deeplabv3plus_v1 \
bash scripts/submit_pipeline.sh
```

For SegFormer after the DeepLabV3+ baseline:

```bash
CONFIG=configs/segformer.yaml \
RUN_NAME=segformer_b0_v1 \
bash scripts/submit_pipeline.sh
```

The CPU setup/download job uses the low-priority `long` queue by default.
Its walltime can be increased when a larger download needs more than 24 hours.
Training uses only the selected `dgx` or `max_dgx` queue:

```bash
# Increase the long-queue walltime when a larger download needs over 24 h.
DOWNLOAD_WALLTIME=48:00:00 \
bash scripts/submit_setup_and_download.sh
```

## PBS resources

The pipeline follows the working MANIT/AudioPrism2.0 convention:

```bash
#PBS -q dgx
#PBS -l select=1:ncpus=12:mem=64gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
```

The `#PBS -q dgx` line is the safe default; the launcher overrides it with
`qsub -q max_dgx` when `GPU_QUEUE=max_dgx` is selected.

Change only these resource lines if MANIT changes the allocation policy.

A short optional GPU smoke test is available:

```bash
# dgx (default)
bash scripts/submit_gpu_smoke.sh

# max_dgx
GPU_QUEUE=max_dgx bash scripts/submit_gpu_smoke.sh

qstat -u "$USER"
```

## Dataset and model contract

- Input: Sentinel-2 Level-2A imagery from 2021.
- Bands: B02, B03, B04, B08, B11, B12.
- Labels: ESA WorldCover 2021 v200, aligned to a 10 m UTM grid.
- Classes: other, agriculture, built-up, natural vegetation, water, and
  bare/sparse.
- Baseline: DeepLabV3+ with ResNet-34.
- Comparison: SegFormer-B0 on the exact same geographic splits.
- Metrics: mIoU, macro F1, per-class IoU/F1, agriculture IoU/F1, agriculture
  area error, and confusion matrix.

The complete remapping and raster contract is documented in
[data/README.md](data/README.md).

The three included AOIs are a pipeline pilot:

- Bhopal: train
- Indore: validation
- Jabalpur: test

For the final research experiment, add several non-touching train regions and
multiple validation/test regions. Never randomly split neighbouring patches.

WorldCover is a derived land-cover product rather than perfect human ground
truth. Final claims should also use a smaller independently inspected test set.

## Model acceptance targets

These are initial project targets, not universal thresholds:

| Metric | Target |
|---|---:|
| mIoU | >= 0.70 |
| Agriculture IoU | >= 0.80 |
| Agriculture F1 | >= 0.85 |
| Agriculture area error | <= 10% |
| Geographic split | No AOI overlap |

## Repository setup

```bash
cd "$HOME"
git clone https://github.com/AntimonyTrisulfide/AstraGuard-LandClassifier.git
cd AstraGuard-LandClassifier
```

## Authoritative references

- [ESA WorldCover data access](https://esa-worldcover.org/en/data-access)
- [WorldCover 2021 v200](https://doi.org/10.5281/zenodo.7254221)
- [Planetary Computer STAC API](https://planetarycomputer.microsoft.com/docs/reference/stac/)
- [Hugging Face SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)
