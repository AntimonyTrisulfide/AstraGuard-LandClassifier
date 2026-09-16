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
- downloads 36 geographically distributed Madhya Pradesh AOIs directly on HPC;
- enforces a 50 GiB raw-data safety ceiling;
- skips any region that is already complete.

Its persistent log is written under:

```text
$HOME/AstraGuard-LandClassifier-data/logs/setup_download_<job-id>.log
```

Wait for this job to finish successfully before entering the second command.
If a previous attempt stopped while compiling a Python dependency, pull the
latest version and submit this same command again. The bootstrap repairs the
existing environment, restricts third-party dependencies to compatible binary
wheels, pins the SegFormer-compatible Transformers 4 API, and skips completed
downloads. Setup is considered complete only after both pretrained models have
been cached successfully.

Default persistent locations:

```text
$HOME/AstraGuard-LandClassifier-data/raw
$HOME/.cache/astraguard-landcover
```

The default [AOI manifest](configs/mp_aois.tsv) contains 26 training, five
validation, and five held-out test regions. Each region covers `0.5° × 0.5°`;
the complete 36-AOI dataset occupied about 7 GiB on MANIT. The downloader prints cumulative
usage after every AOI and stops before the configured 50 GiB limit.

### Optional larger Madhya Pradesh dataset

The separate [216-AOI manifest](configs/mp_aois_expanded.tsv) retains all 36
default AOIs and adds 180 state-centered `0.5° × 0.5°` windows: 156 train,
30 validation, and 30 test AOIs in total. Their split footprints do not
overlap. The additional windows were generated against the
[geoBoundaries India ADM1 boundary](https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/)
(DataMeet India community / Election Commission of India, CC BY 2.5 IN); the
reproducible generator is [scripts/generate_mp_expanded_manifest.py](scripts/generate_mp_expanded_manifest.py).

At similar compression, 216 AOIs should be around 42 GiB, but file size is
not guaranteed: the downloader's 50 GiB raw-data limit remains active. The
windows overlap within a split, so the unique mapped area is about 3.3 times
the baseline, not six times. This is still at Sentinel-2's native 10 m
resolution.

To download the extra AOIs in the CPU `long` queue while a **default** GPU
job is queued, leave `configs/mp_aois.tsv` unchanged and submit:

```bash
cd "$HOME/AstraGuard-LandClassifier"
AOI_MANIFEST=configs/mp_aois_expanded.tsv \
DOWNLOAD_WALLTIME=48:00:00 \
bash scripts/submit_setup_and_download.sh
```

Completed AOIs are skipped, so the first 36 do not download again. The
already-queued default GPU job continues to use the original 36-AOI manifest;
it will **not** automatically switch to the expanded data. After the CPU job
prints `All 216 AOIs are complete`, submit a separate GPU experiment:

```bash
AOI_MANIFEST=configs/mp_aois_expanded.tsv \
RUN_NAME=deeplabv3plus_mp216_v1 \
GPU_QUEUE=dgx \
bash scripts/submit_pipeline.sh
```

Use `GPU_QUEUE=max_dgx` if preferred. The expanded preprocessing/HDF5 scratch
usage and 60-epoch training time may exceed the current 24-hour GPU request;
use the 36-AOI run as a benchmark before requesting or launching a full
expanded run.

### Second command: preprocess, train, and evaluate

This command can be entered even when no GPU is currently free:

```bash
cd "$HOME/AstraGuard-LandClassifier"
# Default GPU queue: dgx
bash scripts/submit_pipeline.sh

# Or select max_dgx
GPU_QUEUE=max_dgx bash scripts/submit_pipeline.sh

# The MANIT launcher defaults to gpu2. Override GPU_HOST only if another
# GPU node has been verified as healthy.
GPU_QUEUE=dgx bash scripts/submit_pipeline.sh

# Wait until an active job occupying the selected host has ended.
GPU_HOST=gpu2 GPU_AFTER_JOB_ID=35905.hpc.local \
GPU_MEM=24gb GPU_QUEUE=dgx bash scripts/submit_pipeline.sh
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

The dataset manifest and raw-data limit can be overridden explicitly:

```bash
AOI_MANIFEST=configs/mp_aois.tsv MAX_RAW_GB=50 \
bash scripts/submit_setup_and_download.sh
```

## PBS resources

The pipeline follows the working MANIT/AudioPrism2.0 convention:

```bash
#PBS -q dgx
#PBS -l select=1:ncpus=12:mem=24gb:ngpus=1
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

The default MP manifest contains 36 non-identical city-and-rural AOIs:

- 26 training regions distributed across central, western, and eastern MP;
- five validation regions;
- five untouched test regions: Gwalior, Jhabua, Balaghat, Singrauli, and
  Burhanpur.

The original three `0.2° × 0.2°` pilot directories may remain on disk, but the
manifest-driven preprocessor excludes them. Splits are assigned by whole AOI;
neighbouring image tiles are never randomly distributed across splits.

The output grid remains at a genuine 10 m resolution. This is already the
highest native spatial resolution among the selected Sentinel-2 bands; B11 and
B12 are natively 20 m and are aligned onto the 10 m grid. Setting a smaller
pixel size would only interpolate pixels and would not add real information.

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
