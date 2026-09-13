"""Download or validate every region in an AOI manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aoi_manifest import AOIRecord, load_aoi_manifest
from .download_aoi import download_region

GIB = 1024**3
REQUIRED_REGION_FILES = ("image.tif", "worldcover.tif", "metadata.json")


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def region_is_complete(
    record: AOIRecord,
    output_root: Path,
    start_date: str,
    end_date: str,
) -> bool:
    region_dir = output_root / record.region_id
    if any(not (region_dir / name).is_file() for name in REQUIRED_REGION_FILES):
        return False
    try:
        with (region_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    stored_bbox = tuple(float(value) for value in metadata.get("bbox_epsg4326", []))
    return (
        metadata.get("region_id") == record.region_id
        and metadata.get("split") == record.split
        and stored_bbox == record.bbox
        and metadata.get("start_date") == start_date
        and metadata.get("end_date") == end_date
    )


def validate_regions(
    records: list[AOIRecord],
    output_root: Path,
    start_date: str,
    end_date: str,
) -> list[str]:
    return [
        record.region_id
        for record in records
        if not region_is_complete(record, output_root, start_date, end_date)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--max-scenes", type=int, default=12)
    parser.add_argument("--max-raw-gb", type=float, default=50.0)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_raw_gb <= 1:
        raise SystemExit("max-raw-gb must be greater than 1 GiB")
    records = load_aoi_manifest(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.check_only:
        missing = validate_regions(
            records, args.output_root, args.start_date, args.end_date
        )
        if missing:
            print("Missing or mismatched AOIs:")
            for region_id in missing:
                print(f"  {region_id}")
            raise SystemExit(1)
        print(f"Validated {len(records)} complete AOIs from {args.manifest}")
        return

    limit_bytes = int(args.max_raw_gb * GIB)
    reserved_bytes = GIB
    for index, record in enumerate(records, start=1):
        if region_is_complete(record, args.output_root, args.start_date, args.end_date):
            print(f"[{index}/{len(records)}] Complete; skipping {record.region_id}")
            continue
        used_bytes = directory_size(args.output_root)
        if used_bytes + reserved_bytes >= limit_bytes:
            raise RuntimeError(
                f"Raw-data safety limit reached: {used_bytes / GIB:.2f} GiB used, "
                f"{args.max_raw_gb:.2f} GiB limit"
            )
        print(f"[{index}/{len(records)}] Downloading {record.region_id} ({record.split})")
        download_region(
            region_id=record.region_id,
            split=record.split,
            bbox=list(record.bbox),
            start_date=args.start_date,
            end_date=args.end_date,
            output_root=args.output_root,
            max_cloud=args.max_cloud,
            max_scenes=args.max_scenes,
        )
        if not region_is_complete(record, args.output_root, args.start_date, args.end_date):
            raise RuntimeError(f"Download did not complete: {record.region_id}")
        used_bytes = directory_size(args.output_root)
        if used_bytes > limit_bytes:
            raise RuntimeError(
                f"Raw data exceeded the limit: {used_bytes / GIB:.2f} GiB used, "
                f"{args.max_raw_gb:.2f} GiB limit"
            )
        print(f"Raw data currently uses {used_bytes / GIB:.2f} GiB")

    missing = validate_regions(records, args.output_root, args.start_date, args.end_date)
    if missing:
        raise RuntimeError(f"Incomplete AOIs after download: {', '.join(missing)}")
    used_bytes = directory_size(args.output_root)
    print(f"All {len(records)} AOIs are complete ({used_bytes / GIB:.2f} GiB raw data)")


if __name__ == "__main__":
    main()
