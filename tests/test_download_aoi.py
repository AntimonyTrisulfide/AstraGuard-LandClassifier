import unittest

import numpy as np

from astraguard_landcover.download_aoi import FLOAT32_FILL_VALUE


class StackstacContractTest(unittest.TestCase):
    def test_nan_fill_value_matches_float32_stack_dtype(self) -> None:
        self.assertIsInstance(FLOAT32_FILL_VALUE, np.float32)
        self.assertTrue(np.isnan(FLOAT32_FILL_VALUE))


if __name__ == "__main__":
    unittest.main()
