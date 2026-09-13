import unittest

import numpy as np

from astraguard_landcover.classes import IGNORE_INDEX, remap_worldcover


class ClassMappingTest(unittest.TestCase):
    def test_worldcover_mapping(self) -> None:
        source = np.array([[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]])
        expected = np.array(
            [[IGNORE_INDEX, 3, 3, 3, 1, 2, 5, 0, 4, 0, 3, 3]], dtype=np.uint8
        )
        np.testing.assert_array_equal(remap_worldcover(source), expected)


if __name__ == "__main__":
    unittest.main()

