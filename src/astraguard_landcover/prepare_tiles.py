"""Convert aligned region GeoTIFF pairs into split-level HDF5 tile stores."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .classes import (
    BAND_NAMES,
    CLASS_NAMES,
    IGNORE_INDEX,
    remap_worldcover,
)
from .utils import write_json


class H5SplitWriter:
    def __init__(self, path: Path, tile_size: int, buffer_size: int = 128) -> None:
        self.path = path
        self.tile_size = tile_size
        self.buffer_size = buffer_size
        self.handle = h5py.File(path, "w", libver="latest")
        self.images = self.handle.create_dataset(
            "images",
            shape=(0, len(BAND_NAMES), tile_size, tile_size),
            maxshape=(None, len(BAND_NAMES), tile_size, tile_size),
            chunks=(1, len(BAND_NAMES), tile_size, tile_size),
            dtype=np.uint16,
            compression="lzf",
        )
        self.masks = self.handle.create_dataset(
            "masks",
            shape=(0, tile_size, tile_size),
            maxshape=(None, tile_size, tile_size),
            chunks=(1, tile_size, tile_size),
            dtype=np.uint8,
            compression="lzf",
        )
        text_dtype = h5py.string_dtype(encoding="utf-8")
        self.keys = self.handle.create_dataset(
            "keys", shape=(0,), maxshape=(None,), dtype=text_dtype
        )
        self.regions = self.handle.create_dataset(
            "regions", shape=(0,), maxshape=(None,), dtype=text_dtype
        )
        self.crs = self.handle.create_dataset(
            "crs", shape=(0,), maxshape=(None,), dtype=text_dtype
        )
        self.transforms = self.handle.create_dataset(
            "transforms", shape=(0, 6), maxshape=(None, 6), dtype=np.float64
        )
        self._buffer: list[tuple[np.ndarray, np.ndarray, str, str, str, tuple[float, ...]]] = []
        self.count = 0

    def append(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        key: str,
        region: str,
        crs: str,
        transform: tuple[float, ...],
    ) -> None:
        self._buffer.append((image, mask, key, region, crs, transform))
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        start = self.count
        end = start + len(self._buffer)
        for dataset in (
            self.images,
            self.masks,
            self.keys,
            self.regions,
            self.crs,
            self.transforms,
        ):
            dataset.resize(end, axis=0)
        self.images[start:end] = np.stack([row[0] for row in self._buffer])
        self.masks[start:end] = np.stack([row[1] for row in self._buffer])
        self.keys[start:end] = [row[2] for row in self._buffer]
        self.regions[start:end] = [row[3] for row in self._buffer]
        self.crs[start:end] = [row[4] for row in self._buffer]
        self.transforms[start:end] = np.asarray([row[5] for row in self._buffer])
        self.count = end
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self.handle.attrs["band_names"] = json.dumps(BAND_NAMES)
        self.handle.attrs["class_names"] = json.dumps(CLASS_NAMES)
        self.handle.attrs["ignore_index"] = IGNORE_INDEX
        self.handle.attrs["tile_size"] = self.tile_size
        self.handle.flush()
        self.handle.swmr_mode = True
        self.handle.close()


def _load_region_metadata(region_dir: Path) -> dict[str, Any]:
    metadata_path = region_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    split = str(metadata.get("split", "")).lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"{metadata_path}: split must be train, val, or test")
    return metadata


def _prepare(
    raw_dir: Path,
    output_dir: Path,
    tile_size: int,
    stride: int,
    min_valid_fraction: float,
    scale_divisor: float,
) -> dict[str, Any]:
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError("Tile preparation requires: pip install -e '.[geo]'") from exc

    region_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if not region_dirs:
        raise FileNotFoundError(f"No region directories found in {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        split: H5SplitWriter(output_dir / f"{split}.h5", tile_size)
        for split in ("train", "val", "test")
    }
    counts = Counter()
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    band_sum = np.zeros(len(BAND_NAMES), dtype=np.float64)
    band_squared_sum = np.zeros(len(BAND_NAMES), dtype=np.float64)
    band_pixel_count = np.zeros(len(BAND_NAMES), dtype=np.int64)
    region_manifest: list[dict[str, Any]] = []

    try:
        for region_dir in region_dirs:
            metadata = _load_region_metadata(region_dir)
            split = str(metadata["split"]).lower()
            image_path = region_dir / "image.tif"
            label_path = region_dir / "worldcover.tif"
            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"{region_dir} must contain image.tif and worldcover.tif"
                )
            accepted = 0
            rejected = 0
            with rasterio.open(image_path) as image_source, rasterio.open(
                label_path
            ) as label_source:
                if image_source.count != len(BAND_NAMES):
                    raise ValueError(
                        f"{image_path} has {image_source.count} bands; "
                        f"expected {len(BAND_NAMES)}"
                    )
                same_grid = (
                    image_source.crs == label_source.crs
                    and image_source.width == label_source.width
                    and image_source.height == label_source.height
                    and image_source.transform.almost_equals(label_source.transform)
                )
                if not same_grid:
                    raise ValueError(
                        f"Image/label grids differ in {region_dir}; reproject labels "
                        "to the image grid before tiling."
                    )
                for row in range(0, image_source.height - tile_size + 1, stride):
                    for column in range(0, image_source.width - tile_size + 1, stride):
                        window = Window(column, row, tile_size, tile_size)
                        image = image_source.read(window=window, out_dtype="float32")
                        raw_mask = label_source.read(1, window=window)
                        finite = np.all(np.isfinite(image), axis=0)
                        image_present = np.any(image != 0, axis=0)
                        label_present = raw_mask != 0
                        valid = finite & image_present & label_present
                        if float(valid.mean()) < min_valid_fraction:
                            rejected += 1
                            continue

                        mask = remap_worldcover(raw_mask)
                        mask[~valid] = IGNORE_INDEX
                        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                        image = np.clip(np.rint(image), 0, np.iinfo(np.uint16).max).astype(
                            np.uint16
                        )
                        key = f"{region_dir.name}_r{row:07d}_c{column:07d}"
                        affine = image_source.window_transform(window)
                        transform = tuple(float(value) for value in affine)[:6]
                        writers[split].append(
                            image,
                            mask,
                            key,
                            region_dir.name,
                            image_source.crs.to_string() if image_source.crs else "",
                            transform,
                        )
                        accepted += 1
                        counts[split] += 1

                        if split == "train":
                            valid_train = mask != IGNORE_INDEX
                            train_values = image[:, valid_train].astype(np.float64) / scale_divisor
                            band_sum += train_values.sum(axis=1)
                            band_squared_sum += np.square(train_values).sum(axis=1)
                            band_pixel_count += train_values.shape[1]
                            class_counts += np.bincount(
                                mask[valid_train], minlength=len(CLASS_NAMES)
                            )
            region_manifest.append(
                {
                    "region": region_dir.name,
                    "split": split,
                    "accepted_tiles": accepted,
                    "rejected_tiles": rejected,
                    "source_metadata": metadata,
                }
            )
    finally:
        for writer in writers.values():
            writer.close()

    missing_splits = [split for split in ("train", "val", "test") if counts[split] == 0]
    if missing_splits:
        raise ValueError(f"No accepted tiles for split(s): {', '.join(missing_splits)}")
    if np.any(band_pixel_count == 0):
        raise ValueError("No valid training pixels available for band statistics")

    mean = band_sum / band_pixel_count
    variance = band_squared_sum / band_pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    total_labels = int(class_counts.sum())
    frequencies = class_counts / max(total_labels, 1)
    present = class_counts > 0
    weights = np.zeros_like(frequencies, dtype=np.float64)
    weights[present] = 1.0 / np.sqrt(frequencies[present])
    if np.any(present):
        weights[present] /= weights[present].mean()
        weights[present] = np.clip(weights[present], 0.25, 4.0)

    stats = {
        "band_names": list(BAND_NAMES),
        "band_mean": mean.tolist(),
        "band_std": std.tolist(),
        "scale_divisor": scale_divisor,
        "class_names": list(CLASS_NAMES),
        "class_pixel_counts": class_counts.tolist(),
        "class_frequencies": frequencies.tolist(),
        "class_weights": weights.tolist(),
        "ignore_index": IGNORE_INDEX,
    }
    manifest = {
        "tile_size": tile_size,
        "stride": stride,
        "min_valid_fraction": min_valid_fraction,
        "band_names": list(BAND_NAMES),
        "class_names": list(CLASS_NAMES),
        "split_tile_counts": dict(counts),
        "regions": region_manifest,
    }
    write_json(output_dir / "stats.json", stats)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--scale-divisor", type=float, default=10000.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tile_size <= 0 or args.stride <= 0:
        raise SystemExit("tile-size and stride must be positive")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise SystemExit("min-valid-fraction must be between 0 and 1")
    manifest = _prepare(
        args.raw_dir,
        args.output_dir,
        args.tile_size,
        args.stride,
        args.min_valid_fraction,
        args.scale_divisor,
    )
    print(json.dumps(manifest["split_tile_counts"], indent=2))


if __name__ == "__main__":
    main()

