"""Sliding-window GeoTIFF inference with georeferenced mask and area statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .classes import BAND_NAMES, CLASS_COLORS, CLASS_NAMES, IGNORE_INDEX
from .models import build_model
from .utils import select_device, write_json


def _starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def predict_raster(
    *,
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    tile_size: int,
    overlap: int,
    batch_size: int,
    max_pixels: int,
) -> dict[str, Any]:
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError("GeoTIFF prediction requires: pip install -e '.[geo]'") from exc

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile-size")
    device = select_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    stats = checkpoint["stats"]
    model = build_model(config["model"], initialize_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    mean = np.asarray(stats["band_mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["band_std"], dtype=np.float32)[:, None, None]
    scale = float(stats["scale_divisor"])

    with rasterio.open(input_path) as source:
        if source.count != len(BAND_NAMES):
            raise ValueError(
                f"Input has {source.count} bands; expected {len(BAND_NAMES)} in order "
                f"{', '.join(BAND_NAMES)}"
            )
        pixel_count = source.width * source.height
        if pixel_count > max_pixels:
            raise ValueError(
                f"Raster has {pixel_count:,} pixels, above --max-pixels={max_pixels:,}. "
                "Process it as smaller AOIs or explicitly raise the limit."
            )
        probability_sum = np.zeros(
            (len(CLASS_NAMES), source.height, source.width), dtype=np.float32
        )
        prediction_count = np.zeros((source.height, source.width), dtype=np.uint16)
        step = tile_size - overlap
        rows = _starts(source.height, tile_size, step)
        columns = _starts(source.width, tile_size, step)
        pending_images: list[np.ndarray] = []
        pending_locations: list[tuple[int, int, int, int]] = []

        @torch.no_grad()
        def flush() -> None:
            if not pending_images:
                return
            batch = torch.from_numpy(np.stack(pending_images)).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                probabilities = torch.softmax(model(batch), dim=1)
            probabilities_np = probabilities.float().cpu().numpy()
            for probabilities_tile, (row, column, height, width) in zip(
                probabilities_np, pending_locations, strict=True
            ):
                probability_sum[:, row : row + height, column : column + width] += (
                    probabilities_tile[:, :height, :width]
                )
                prediction_count[row : row + height, column : column + width] += 1
            pending_images.clear()
            pending_locations.clear()

        for row in rows:
            for column in columns:
                height = min(tile_size, source.height - row)
                width = min(tile_size, source.width - column)
                raw = source.read(
                    window=Window(column, row, width, height), out_dtype="float32"
                )
                padded = np.zeros(
                    (len(BAND_NAMES), tile_size, tile_size), dtype=np.float32
                )
                normalized = (raw / scale - mean) / std
                padded[:, :height, :width] = normalized
                pending_images.append(padded)
                pending_locations.append((row, column, height, width))
                if len(pending_images) >= batch_size:
                    flush()
        flush()

        averaged = probability_sum / np.maximum(prediction_count[None, ...], 1)
        prediction = averaged.argmax(axis=0).astype(np.uint8)
        valid_raster = source.dataset_mask() != 0
        prediction[~valid_raster] = IGNORE_INDEX
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=IGNORE_INDEX,
            compress="DEFLATE",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(prediction, 1)
            destination.write_colormap(1, CLASS_COLORS)
            destination.set_band_description(1, "land_cover_class_id")

        counts = np.bincount(
            prediction[prediction != IGNORE_INDEX], minlength=len(CLASS_NAMES)
        )
        projected = bool(source.crs and source.crs.is_projected)
        linear_units = source.crs.linear_units if projected else None
        uses_metres = str(linear_units).lower() in {
            "metre",
            "metres",
            "meter",
            "meters",
        }
        area_available = projected and uses_metres
        pixel_area_square_m = (
            abs(
                source.transform.a * source.transform.e
                - source.transform.b * source.transform.d
            )
            if area_available
            else None
        )
        classes = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            classes[class_name] = {
                "class_id": class_id,
                "pixels": int(counts[class_id]),
                "area_km2": (
                    float(counts[class_id] * pixel_area_square_m / 1_000_000)
                    if pixel_area_square_m is not None
                    else None
                ),
            }
        result = {
            "input": str(input_path),
            "checkpoint": str(checkpoint_path),
            "output_mask": str(output_path),
            "crs": source.crs.to_string() if source.crs else None,
            "pixel_area_square_m": pixel_area_square_m,
            "area_available": area_available,
            "crs_linear_units": linear_units,
            "classes": classes,
        }
    write_json(output_path.with_suffix(".json"), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=100_000_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = predict_raster(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        max_pixels=args.max_pixels,
    )
    agriculture = result["classes"]["agriculture"]
    print(
        f"Wrote {result['output_mask']}; agriculture area = "
        f"{agriculture['area_km2']} km^2"
    )


if __name__ == "__main__":
    main()
