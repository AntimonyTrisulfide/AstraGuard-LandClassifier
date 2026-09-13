import unittest

import torch

from astraguard_landcover.metrics import SegmentationMetrics


class MetricsTest(unittest.TestCase):
    def test_known_confusion_and_ignore(self) -> None:
        prediction = torch.tensor([[[1, 1], [2, 0]]])
        target = torch.tensor([[[1, 2], [2, 255]]])
        metrics = SegmentationMetrics(num_classes=6, ignore_index=255)
        metrics.update(prediction, target)
        result = metrics.compute()
        self.assertAlmostEqual(result["overall_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["miou"], 0.5)
        self.assertAlmostEqual(result["per_class"]["agriculture"]["iou"], 0.5)
        self.assertAlmostEqual(result["agriculture_area_error_percent"], 100.0)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 3)


if __name__ == "__main__":
    unittest.main()

