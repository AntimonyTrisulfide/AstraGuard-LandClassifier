"""Print a reproducible, state-centered AOI expansion as TSV.

Uses the Madhya Pradesh ADM1 boundary from geoBoundaries (DataMeet India
community / Election Commission of India, CC BY 2.5 IN). The boundary is
fetched only when regenerating the manifest; training uses the committed TSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

BOUNDARY_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
    "releaseData/gbOpen/IND/ADM1/"
    "geoBoundaries-IND-ADM1_simplified.geojson"
)
FIELDS = ("region_id", "split", "west", "south", "east", "north")
BASE_WEST = 73.75
BASE_SOUTH = 20.75
GRID_STEP = 0.25
WINDOW_SIZE = 0.5
TARGET_COUNTS = {"train": 156, "val": 30, "test": 30}


@dataclass(frozen=True)
class Region:
    region_id: str
    split: str
    west: float
    south: float
    east: float
    north: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2, (self.south + self.north) / 2)


def _inside_ring(point: tuple[float, float], ring: list[list[float]]) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = ring[-1]
    for current_x, current_y in ring:
        if (current_y > y) != (previous_y > y):
            crossing_x = current_x + (
                (y - current_y) * (previous_x - current_x)
                / (previous_y - current_y)
            )
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _inside_polygon(point: tuple[float, float], rings: list) -> bool:
    return _inside_ring(point, rings[0]) and not any(
        _inside_ring(point, hole) for hole in rings[1:]
    )


def _state_polygons(geojson: dict) -> list[list]:
    matching = [
        feature for feature in geojson["features"]
        if feature["properties"].get("shapeName") == "Madhya Pradesh"
    ]
    if len(matching) != 1:
        raise ValueError("Expected exactly one Madhya Pradesh ADM1 feature")
    geometry = matching[0]["geometry"]
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError(f"Unexpected Madhya Pradesh geometry: {geometry['type']}")


def _inside_state(point: tuple[float, float], polygons: list[list]) -> bool:
    return any(_inside_polygon(point, rings) for rings in polygons)


def _overlaps(left: Region, right: Region) -> bool:
    return (
        max(left.west, right.west) < min(left.east, right.east)
        and max(left.south, right.south) < min(left.north, right.north)
    )


def _distance_squared(left: tuple[float, float], right: tuple[float, float]) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _read_base(path: Path) -> list[Region]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected columns in {path}")
        return [
            Region(
                row["region_id"], row["split"],
                *(float(row[name]) for name in FIELDS[2:]),
            )
            for row in reader
        ]


def _candidate_regions(polygons: list[list], base: list[Region]) -> list[Region]:
    existing_boxes = {(r.west, r.south, r.east, r.north) for r in base}
    candidates = []
    for x_index in range(38):
        for y_index in range(27):
            west = BASE_WEST + x_index * GRID_STEP
            south = BASE_SOUTH + y_index * GRID_STEP
            candidate = Region(
                f"mp_grid_x{x_index:02d}_y{y_index:02d}", "", west, south,
                west + WINDOW_SIZE, south + WINDOW_SIZE,
            )
            if (
                _inside_state(candidate.center, polygons)
                and (candidate.west, candidate.south, candidate.east, candidate.north)
                not in existing_boxes
            ):
                candidates.append(candidate)
    return candidates


def _feasible(candidate: Region, split: str, selected: list[Region]) -> bool:
    return all(
        other.split == split or not _overlaps(candidate, other)
        for other in selected
    )


def _add_holdouts(
    split: str, selected: list[Region], candidates: list[Region],
    base: list[Region], target: int,
) -> None:
    anchors = [region for region in base if region.split == split]
    while sum(region.split == split for region in selected) < target:
        eligible = [
            candidate for candidate in candidates
            if _feasible(candidate, split, selected)
            and min(_distance_squared(candidate.center, a.center) for a in anchors)
            <= 1.5 ** 2
        ]
        if not eligible:
            raise ValueError(f"Not enough non-leaking {split} candidates")
        same_split = [region for region in selected if region.split == split]
        # Prioritize new geographic coverage while keeping the held-out
        # regions clustered around their original split anchors.
        best = max(
            eligible,
            key=lambda candidate: (
                min(_distance_squared(candidate.center, r.center) for r in same_split),
                -min(_distance_squared(candidate.center, a.center) for a in anchors),
                candidate.region_id,
            ),
        )
        selected.append(Region(best.region_id + f"_{split}", split, best.west,
                               best.south, best.east, best.north))
        candidates.remove(best)


def expand(base: list[Region], polygons: list[list]) -> list[Region]:
    selected = list(base)
    candidates = _candidate_regions(polygons, base)
    for split in ("val", "test"):
        _add_holdouts(split, selected, candidates, base, TARGET_COUNTS[split])
    while sum(region.split == "train" for region in selected) < TARGET_COUNTS["train"]:
        eligible = [
            candidate for candidate in candidates
            if _feasible(candidate, "train", selected)
        ]
        if not eligible:
            raise ValueError("Not enough non-leaking train candidates")
        training = [region for region in selected if region.split == "train"]
        best = max(
            eligible,
            key=lambda candidate: (
                min(_distance_squared(candidate.center, r.center) for r in training),
                candidate.region_id,
            ),
        )
        selected.append(Region(best.region_id + "_train", "train", best.west,
                               best.south, best.east, best.north))
        candidates.remove(best)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("configs/mp_aois.tsv"))
    parser.add_argument("--boundary", type=Path,
                        help="Use a local geoBoundaries GeoJSON instead of downloading it")
    args = parser.parse_args()
    if args.boundary:
        geojson = json.loads(args.boundary.read_text(encoding="utf-8"))
    else:
        with urlopen(BOUNDARY_URL, timeout=60) as response:
            geojson = json.load(response)
    regions = expand(_read_base(args.base), _state_polygons(geojson))
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(FIELDS)
    for region in regions:
        writer.writerow((region.region_id, region.split,
                         f"{region.west:.2f}", f"{region.south:.2f}",
                         f"{region.east:.2f}", f"{region.north:.2f}"))


if __name__ == "__main__":
    main()
