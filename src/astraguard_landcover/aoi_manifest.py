"""Parsing and validation for tab-separated AOI manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AOIRecord:
    region_id: str
    split: str
    bbox: tuple[float, float, float, float]


def load_aoi_manifest(path: Path) -> list[AOIRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"AOI manifest does not exist: {path}")

    records: list[AOIRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"region_id", "split", "west", "south", "east", "north"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                f"{path} must have exactly these tab-separated columns: "
                f"{', '.join(sorted(expected))}"
            )
        for line_number, row in enumerate(reader, start=2):
            region_id = row["region_id"].strip()
            split = row["split"].strip().lower()
            if not region_id:
                raise ValueError(f"{path}:{line_number}: region_id is empty")
            if region_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate region_id {region_id}")
            if split not in {"train", "val", "test"}:
                raise ValueError(
                    f"{path}:{line_number}: split must be train, val, or test"
                )
            try:
                bbox = tuple(float(row[name]) for name in ("west", "south", "east", "north"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid bbox") from exc
            west, south, east, north = bbox
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError(f"{path}:{line_number}: invalid EPSG:4326 bbox")
            records.append(AOIRecord(region_id=region_id, split=split, bbox=bbox))
            seen.add(region_id)

    if not records:
        raise ValueError(f"AOI manifest is empty: {path}")
    missing_splits = {"train", "val", "test"} - {record.split for record in records}
    if missing_splits:
        raise ValueError(f"AOI manifest has no rows for: {', '.join(sorted(missing_splits))}")
    return records
