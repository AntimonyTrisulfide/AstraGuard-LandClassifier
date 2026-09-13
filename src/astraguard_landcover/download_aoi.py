"""Build one aligned Sentinel-2/WorldCover region pair from public STAC data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .classes import BAND_NAMES
from .utils import write_json

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BAD_SCL_VALUES = (0, 1, 3, 8, 9, 10, 11)


def _utm_epsg(longitude: float, latitude: float) -> int:
    zone = int(math.floor((longitude + 180.0) / 6.0) + 1)
    zone = min(max(zone, 1), 60)
    return (32600 if latitude >= 0 else 32700) + zone


def _query_items(
    catalog: Any,
    *,
    collection: str,
    bbox: list[float],
    datetime_range: str,
    query: dict[str, Any] | None = None,
) -> list[Any]:
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query=query,
    )
    return list(search.items())


def _select_monthly_scenes(items: list[Any], max_scenes: int) -> list[Any]:
    """Prefer the least-cloudy scene in each month before taking extra scenes."""
    by_month: dict[int, list[Any]] = {}
    for item in items:
        month = item.datetime.month if item.datetime is not None else 0
        by_month.setdefault(month, []).append(item)
    selected: list[Any] = []
    for month in sorted(by_month):
        candidates = sorted(
            by_month[month],
            key=lambda item: float(item.properties.get("eo:cloud_cover", 100)),
        )
        selected.append(candidates[0])
    if len(selected) < max_scenes:
        selected_ids = {item.id for item in selected}
        remaining = sorted(
            (item for item in items if item.id not in selected_ids),
            key=lambda item: float(item.properties.get("eo:cloud_cover", 100)),
        )
        selected.extend(remaining[: max_scenes - len(selected)])
    selected = selected[:max_scenes]
    return sorted(selected, key=lambda item: str(item.properties.get("datetime", "")))


def download_region(
    *,
    region_id: str,
    split: str,
    bbox: list[float],
    start_date: str,
    end_date: str,
    output_root: Path,
    max_cloud: float,
    max_scenes: int,
) -> dict[str, Any]:
    try:
        import planetary_computer
        import rioxarray  # noqa: F401 - registers the .rio accessor
        import stackstac
        from pystac_client import Client
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise ImportError("AOI download requires: pip install -e '.[geo]'") from exc

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be west south east north in EPSG:4326")
    epsg = _utm_epsg((west + east) / 2.0, (south + north) / 2.0)
    catalog = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    sentinel_items = _query_items(
        catalog,
        collection="sentinel-2-l2a",
        bbox=bbox,
        datetime_range=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    sentinel_items = _select_monthly_scenes(sentinel_items, max_scenes)
    if not sentinel_items:
        raise RuntimeError("No Sentinel-2 scenes matched the AOI/date/cloud filters")

    stack_arguments = {
        "epsg": epsg,
        "resolution": 10,
        "bounds_latlon": bbox,
        "chunksize": 2048,
        "rescale": False,
    }
    reflectance = stackstac.stack(
        sentinel_items,
        assets=list(BAND_NAMES),
        dtype="float32",
        fill_value=np.nan,
        resampling=Resampling.bilinear,
        **stack_arguments,
    )
    scl = stackstac.stack(
        sentinel_items,
        assets=["SCL"],
        dtype="float32",
        fill_value=np.nan,
        resampling=Resampling.nearest,
        **stack_arguments,
    ).sel(band="SCL")
    valid = ~scl.isin(BAD_SCL_VALUES)
    composite = reflectance.where(valid).median(dim="time", skipna=True)
    composite = composite.fillna(0).clip(min=0, max=10000).round().astype("uint16")
    composite = composite.rio.write_crs(epsg).rio.write_nodata(0)

    worldcover_items = _query_items(
        catalog,
        collection="esa-worldcover",
        bbox=bbox,
        datetime_range="2021-01-01/2021-12-31",
    )
    if not worldcover_items:
        raise RuntimeError("No ESA WorldCover 2021 tiles matched the AOI")
    missing_map = [item.id for item in worldcover_items if "map" not in item.assets]
    if missing_map:
        available = sorted(worldcover_items[0].assets)
        raise RuntimeError(
            f"WorldCover STAC item has no 'map' asset. Available assets: {available}"
        )
    worldcover = stackstac.stack(
        worldcover_items,
        assets=["map"],
        epsg=epsg,
        resolution=10,
        bounds_latlon=bbox,
        dtype="uint8",
        fill_value=0,
        rescale=False,
        resampling=Resampling.nearest,
        chunksize=2048,
    ).max(dim="time").sel(band="map")
    worldcover = worldcover.rio.write_crs(epsg).rio.write_nodata(0)
    reference = composite.isel(band=0, drop=True)
    worldcover = worldcover.rio.reproject_match(reference, resampling=Resampling.nearest)
    worldcover = worldcover.fillna(0).astype("uint8")

    region_dir = output_root / region_id
    region_dir.mkdir(parents=True, exist_ok=True)
    image_path = region_dir / "image.tif"
    label_path = region_dir / "worldcover.tif"
    composite.rio.to_raster(
        image_path,
        driver="GTiff",
        tiled=True,
        compress="DEFLATE",
        predictor=2,
        BIGTIFF="IF_SAFER",
    )
    worldcover.rio.to_raster(
        label_path,
        driver="GTiff",
        tiled=True,
        compress="DEFLATE",
        BIGTIFF="IF_SAFER",
    )
    metadata = {
        "region_id": region_id,
        "split": split,
        "bbox_epsg4326": bbox,
        "output_epsg": epsg,
        "start_date": start_date,
        "end_date": end_date,
        "worldcover_year": 2021,
        "max_cloud_percent": max_cloud,
        "sentinel_scene_ids": [item.id for item in sentinel_items],
        "sentinel_scene_cloud_percent": [
            item.properties.get("eo:cloud_cover") for item in sentinel_items
        ],
        "band_names": list(BAND_NAMES),
        "bad_scl_values": list(BAD_SCL_VALUES),
        "stac_url": STAC_URL,
    }
    write_json(region_dir / "metadata.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument(
        "--bbox",
        required=True,
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
    )
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--max-scenes", type=int, default=12)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = download_region(
        region_id=args.region_id,
        split=args.split,
        bbox=list(args.bbox),
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
        max_cloud=args.max_cloud,
        max_scenes=args.max_scenes,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
