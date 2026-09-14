import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np

from astraguard_landcover.download_aoi import (
    FLOAT32_FILL_VALUE,
    _remove_existing_sas,
    _sign_selected_assets,
)


class StackstacContractTest(unittest.TestCase):
    def test_nan_fill_value_matches_float32_stack_dtype(self) -> None:
        self.assertIsInstance(FLOAT32_FILL_VALUE, np.float32)
        self.assertTrue(np.isnan(FLOAT32_FILL_VALUE))

    def test_stale_sas_is_removed_before_resigning(self) -> None:
        old_href = (
            "https://example.blob.core.windows.net/data/image.tif?"
            "versionid=123&se=2026-09-13T21%3A43%3A18Z&sig=old"
        )
        unsigned = _remove_existing_sas(old_href)
        self.assertEqual(
            unsigned,
            "https://example.blob.core.windows.net/data/image.tif?versionid=123",
        )
        self.assertEqual(
            _remove_existing_sas("https://example.org/image.tif?se=old"),
            "https://example.org/image.tif?se=old",
        )

    def test_selected_asset_is_resigned_with_fresh_expiry(self) -> None:
        old_href = (
            "https://example.blob.core.windows.net/data/image.tif?"
            "se=2026-09-13T21%3A43%3A18Z&sig=old"
        )
        item = SimpleNamespace(
            id="example-item", assets={"B04": SimpleNamespace(href=old_href)}
        )
        seen = []
        expiry = quote(
            (datetime.now(timezone.utc) + timedelta(hours=1))
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        def signer(href: str) -> str:
            seen.append(href)
            return f"{href}?se={expiry}&sig=new"

        _sign_selected_assets([item], ["B04"], signer)
        self.assertEqual(seen, [old_href.split("?")[0]])
        self.assertIn("sig=new", item.assets["B04"].href)

    def test_non_sas_query_is_preserved_after_resigning(self) -> None:
        href = (
            "https://example.blob.core.windows.net/data/image.tif?"
            "versionid=123&se=2026-09-13T21%3A43%3A18Z&sig=old"
        )
        item = SimpleNamespace(
            id="example-item", assets={"B04": SimpleNamespace(href=href)}
        )
        expiry = quote(
            (datetime.now(timezone.utc) + timedelta(hours=1))
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        _sign_selected_assets(
            [item], ["B04"], lambda bare: f"{bare}?se={expiry}&sig=new"
        )
        self.assertIn("versionid=123", item.assets["B04"].href)
        self.assertNotIn("sig=old", item.assets["B04"].href)

    def test_expired_response_fails_before_raster_read(self) -> None:
        href = "https://example.blob.core.windows.net/data/image.tif"
        item = SimpleNamespace(
            id="example-item", assets={"B04": SimpleNamespace(href=href)}
        )
        with self.assertRaisesRegex(RuntimeError, "expired or near-expiry"):
            _sign_selected_assets(
                [item], ["B04"],
                lambda unsigned: f"{unsigned}?se=2026-09-13T21%3A43%3A18Z&sig=old",
            )

    def test_valid_cached_sas_with_five_minutes_remaining_is_accepted(self) -> None:
        href = "https://example.blob.core.windows.net/data/image.tif"
        item = SimpleNamespace(
            id="example-item", assets={"B04": SimpleNamespace(href=href)}
        )
        expiry = quote(
            (datetime.now(timezone.utc) + timedelta(minutes=5))
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        _sign_selected_assets(
            [item], ["B04"], lambda unsigned: f"{unsigned}?se={expiry}&sig=new"
        )
        self.assertIn("sig=new", item.assets["B04"].href)

    def test_sas_expiring_within_a_minute_is_rejected(self) -> None:
        href = "https://example.blob.core.windows.net/data/image.tif"
        item = SimpleNamespace(
            id="example-item", assets={"B04": SimpleNamespace(href=href)}
        )
        expiry = quote(
            (datetime.now(timezone.utc) + timedelta(seconds=30))
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        with self.assertRaisesRegex(RuntimeError, "expired or near-expiry"):
            _sign_selected_assets(
                [item], ["B04"],
                lambda unsigned: f"{unsigned}?se={expiry}&sig=new",
            )


if __name__ == "__main__":
    unittest.main()
