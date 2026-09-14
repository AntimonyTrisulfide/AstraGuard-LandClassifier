import unittest
from collections import Counter
from pathlib import Path

from astraguard_landcover.aoi_manifest import load_aoi_manifest


class AOIManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cls.records = load_aoi_manifest(project_root / "configs" / "mp_aois.tsv")
        cls.expanded = load_aoi_manifest(
            project_root / "configs" / "mp_aois_expanded.tsv"
        )

    def test_expected_region_and_split_counts(self) -> None:
        self.assertEqual(len(self.records), 36)
        self.assertEqual(
            Counter(record.split for record in self.records),
            {"train": 26, "val": 5, "test": 5},
        )

    def test_regions_are_half_degree_windows(self) -> None:
        for record in self.records:
            west, south, east, north = record.bbox
            self.assertAlmostEqual(east - west, 0.5)
            self.assertAlmostEqual(north - south, 0.5)

    def test_cross_split_regions_do_not_overlap(self) -> None:
        for records in (self.records, self.expanded):
            for index, left in enumerate(records):
                for right in records[index + 1 :]:
                    if left.split == right.split:
                        continue
                    left_west, left_south, left_east, left_north = left.bbox
                    right_west, right_south, right_east, right_north = right.bbox
                    overlaps = (
                        max(left_west, right_west) < min(left_east, right_east)
                        and max(left_south, right_south) < min(left_north, right_north)
                    )
                    self.assertFalse(
                        overlaps, f"{left.region_id} overlaps {right.region_id}"
                    )

    def test_expanded_manifest_preserves_all_original_regions(self) -> None:
        self.assertEqual(len(self.expanded), 216)
        self.assertEqual(
            Counter(record.split for record in self.expanded),
            {"train": 156, "val": 30, "test": 30},
        )
        self.assertEqual(self.expanded[: len(self.records)], self.records)
        self.assertEqual(
            len({record.region_id for record in self.expanded}),
            len(self.expanded),
        )
        for record in self.expanded:
            west, south, east, north = record.bbox
            self.assertAlmostEqual(east - west, 0.5)
            self.assertAlmostEqual(north - south, 0.5)


if __name__ == "__main__":
    unittest.main()
