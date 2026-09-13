# Data contract

Raw data is organized by **geographically disjoint region**. Never put adjacent
tiles from one region into different splits.

```text
data/raw/
  bhopal_train/
    image.tif       # six bands: B02, B03, B04, B08, B11, B12
    worldcover.tif  # ESA WorldCover 2021 v200 class codes
    metadata.json   # includes split=train|val|test
  indore_val/
    ...
  jabalpur_test/
    ...
```

`image.tif` and `worldcover.tif` must have the same CRS, affine transform,
width, and height. Sentinel-2 reflectance is stored on the usual 0--10000
scale. Pixel value 0 is reserved for nodata.

Run `astraguard-prepare` to convert these region rasters into:

```text
data/processed/
  train.h5
  val.h5
  test.h5
  stats.json
  manifest.json
```

HDF5 is deliberate: a split remains one file instead of becoming tens of
thousands of small `.npy` files on the cluster filesystem.

## Fixed band order

| Index | Sentinel-2 band | Native resolution |
|---:|---|---:|
| 0 | B02 (blue) | 10 m |
| 1 | B03 (green) | 10 m |
| 2 | B04 (red) | 10 m |
| 3 | B08 (NIR) | 10 m |
| 4 | B11 (SWIR-1) | 20 m, resampled to 10 m |
| 5 | B12 (SWIR-2) | 20 m, resampled to 10 m |

## WorldCover remapping

| Model id | Model class | WorldCover codes |
|---:|---|---|
| 0 | other | 70 snow/ice, 90 herbaceous wetland |
| 1 | agriculture | 40 cropland |
| 2 | built_up | 50 built-up |
| 3 | natural_vegetation | 10 tree, 20 shrubland, 30 grassland, 95 mangroves, 100 moss/lichen |
| 4 | water | 80 permanent water |
| 5 | bare_or_sparse | 60 bare/sparse vegetation |
| 255 | ignore | WorldCover nodata or invalid imagery |

This mapping is a project design decision, not an ESA taxonomy. Report results
against both the six-class task and the agriculture class separately.

