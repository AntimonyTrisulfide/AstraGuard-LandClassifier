# AstraGuard — Land Cover Segmentation

This repository is Member 1's independent model project. It trains a
multispectral semantic-segmentation model that converts a Sentinel-2 image into
a pixel-level land-cover map and an agriculture mask.

The finalized dataset pairing is:

- **Input:** Sentinel-2 Level-2A imagery from 2021, composited after SCL cloud
  masking.
- **Training labels:** ESA WorldCover 2021 v200 at 10 m.
- **Initial models:** DeepLabV3+ with ResNet-34, followed by SegFormer-B0 on the
  exact same geographic splits.
- **Primary research metrics:** mIoU, agriculture IoU/F1, confusion matrix, and
  agriculture area error.

WorldCover is a derived product, so this first stage measures how well the model
reproduces its taxonomy. The final report must also use a smaller independently
checked test set; do not claim that WorldCover labels are perfect ground truth.

## What is implemented

```text
Public Sentinel-2 L2A STAC scenes + ESA WorldCover 2021
                         |
                  cloud-masked median
                         |
             aligned six-band region GeoTIFFs
                         |
          geographic train / validation / test regions
                         |
       HDF5 tiles (one file per split, no inode explosion)
                         |
             DeepLabV3+ or SegFormer training
                         |
        test metrics + confusion matrix + best checkpoint
                         |
       georeferenced prediction mask + area statistics
```

The fixed input/output definition is in [data/README.md](data/README.md).

## 1. Put the repository on MANIT HPC

Create and push the Git repository from your development machine. Then connect
to MANIT and clone it into your persistent allocation:

```bash
ssh <username>@<manit-login-host>
cd "$HOME"
git clone <your-repository-url> AstraGuard-Land-Cover
cd AstraGuard-Land-Cover
```

Create the Conda environment once. This follows the PBS setup used by the
AudioPrism2.0 MANIT project:

```bash
ENV_NAME=astraguard-landcover \
CACHE_DIR="$HOME/.cache/astraguard-landcover" \
bash scripts/bootstrap_hpc.sh
```

The bootstrap installs the project, runs unit tests, and downloads the two
pretrained model assets while the login node has network access. If downloads
are not allowed there, set `PREFETCH_MODELS=0` and transfer a populated cache
from another machine.

## 2. Storage on MANIT HPC

Keep code, raw regional rasters, final HDF5 splits, checkpoints, and results on
persistent storage. The PBS training job copies `train.h5` and `val.h5` to the
job-local `$TMPDIR` by default, then writes checkpoints and logs back to the
persistent output directory.

Copy [env/hpc.env.example](env/hpc.env.example) to a private HPC path, replace
the placeholders, and source it before running commands:

```bash
source /path/to/astraguard-landcover.env
cd "$PROJECT_ROOT"
```

The PBS files use the same known MANIT convention as AudioPrism2.0: queue
`dgx`, PBS `select` resources, `PBS_O_WORKDIR`, and `$TMPDIR`. Training requests
one GPU, 12 CPUs, 64 GB RAM, and 24 hours. If your allocation has different
limits, change only the `#PBS` resource lines.

## 3. Build geographically separate raw regions

Run downloads on a login/data-transfer node if outbound internet is disabled on
compute nodes. Each invocation creates `image.tif`, `worldcover.tif`, and
`metadata.json` in one region directory.

The commands below are only a small pipeline check. The boxes are deliberately
separated; expand the real experiment to several train regions and multiple
validation/test regions.

```bash
astraguard-download-aoi \
  --region-id bhopal_train --split train \
  --bbox 77.30 23.15 77.50 23.35 \
  --output-root "$RAW_DIR"

astraguard-download-aoi \
  --region-id indore_val --split val \
  --bbox 75.75 22.62 75.95 22.82 \
  --output-root "$RAW_DIR"

astraguard-download-aoi \
  --region-id jabalpur_test --split test \
  --bbox 79.85 23.08 80.05 23.28 \
  --output-root "$RAW_DIR"
```

The downloader selects up to one low-cloud image per month before adding extra
scenes, masks SCL values for nodata/defects/shadow/cloud/cirrus/snow, creates a
2021 median reflectance composite, and resamples B11/B12 to the common 10 m UTM
grid. Dates are fixed to 2021 to match WorldCover 2021.

For a defensible full experiment:

- Use non-touching AOIs and record their polygons before tiling.
- Put entire AOIs in exactly one split.
- Include rural, peri-urban, forested, water-rich, and bare-land conditions.
- Keep the test regions untouched until both models and hyperparameters are
  finalized.
- Have 500–1000 unseen test tiles independently inspected against higher
  resolution imagery or another authoritative reference.

## 4. Create training tiles

For a local/small run:

```bash
astraguard-prepare \
  --raw-dir "$RAW_DIR" \
  --output-dir "$PROCESSED_DIR" \
  --tile-size 256 --stride 256 --min-valid-fraction 0.95
```

Submit preprocessing through PBS:

```bash
bash scripts/submit_prepare.sh
qstat -u "$USER"
```

`prepare.pbs` uses the `dgx` queue but does not request a GPU. If the queue
policy requires `ngpus=1`, or MANIT provides a separate CPU queue, adjust its
single `#PBS -q`/resource line accordingly.

Inspect `manifest.json` and `stats.json`. Confirm every class count, region
assignment, and split size before spending GPU hours. A zero or tiny class count
means the AOIs need to be redesigned.

## 5. Cache pretrained weights once

Compute nodes often have no internet. On a network-enabled node, populate a
persistent cache first:

```bash
export HF_HOME="$CACHE_DIR/huggingface"
export TORCH_HOME="$CACHE_DIR/torch"
"$PYTHON_BIN" scripts/prefetch_models.py \
  configs/deeplabv3plus.yaml configs/segformer.yaml
```

If no HPC node has outbound access, run the same command elsewhere with the same
package versions and copy these two cache directories to the path configured on
the HPC.

## 6. Train in the agreed order

First establish the conventional baseline:

```bash
CONFIG=configs/deeplabv3plus.yaml \
RUN_NAME=deeplabv3plus_v1 \
bash scripts/submit.sh
```

After the data pipeline and metrics are verified, train SegFormer on the exact
same HDF5 files:

```bash
CONFIG=configs/segformer.yaml \
RUN_NAME=segformer_b0_v1 \
bash scripts/submit.sh
```

Each run produces `best.pt`, `last.pt`, `history.csv`, the resolved YAML config,
and validation metrics. Resume an interrupted local run with:

```bash
"$PYTHON_BIN" -m astraguard_landcover.train \
  --config configs/deeplabv3plus.yaml \
  --data-dir "$PROCESSED_DIR" \
  --output-dir "$OUTPUT_DIR/deeplabv3plus_v1" \
  --resume "$OUTPUT_DIR/deeplabv3plus_v1/last.pt"
```

The PBS launcher performs this resume automatically whenever `last.pt` exists.
Monitor it with `qstat -u "$USER"` and the persistent log under
`$OUTPUT_DIR/<run-name>/logs/`.

## 7. Final test and inference

Evaluate the winning validation checkpoint exactly once on the held-out test:

```bash
export CHECKPOINT="$OUTPUT_DIR/deeplabv3plus_v1/best.pt"
bash scripts/submit_evaluate.sh
```

This writes `test/metrics.json` and `test/confusion_matrix.png` beside the
checkpoint.

Run standalone inference on a six-band aligned GeoTIFF:

```bash
astraguard-predict \
  --checkpoint "$CHECKPOINT" \
  --input "$RAW_DIR/bhopal_train/image.tif" \
  --output "$OUTPUT_DIR/demo/bhopal_landcover.tif"
```

The result is a georeferenced class-id GeoTIFF plus a JSON file containing
per-class pixel counts and square-kilometre areas. Area is only reported for a
projected CRS with metre units; the downloader's UTM output satisfies that
condition.

## Model acceptance gate

Treat these as initial project targets, not universal scientific thresholds:

| Metric | Initial target |
|---|---:|
| mIoU | >= 0.70 |
| Agriculture IoU | >= 0.80 |
| Agriculture F1 | >= 0.85 |
| Agriculture area error | <= 10% |
| Geographic test | No train/val/test AOI overlap |

Do not integrate a checkpoint into AstraGuard merely because it has high
training accuracy. The accepted artifact is `best.pt` together with its config,
data manifest, test metrics, input/output contract, and known limitations.

## Authoritative data references

- [ESA WorldCover data access and license](https://esa-worldcover.org/en/data-access)
- [WorldCover 2021 v200 record](https://doi.org/10.5281/zenodo.7254221)
- [Microsoft Planetary Computer STAC API](https://planetarycomputer.microsoft.com/docs/reference/stac/)
- [Hugging Face SegFormer documentation](https://huggingface.co/docs/transformers/model_doc/segformer)
