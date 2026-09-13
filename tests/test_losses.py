import unittest

import torch

from astraguard_landcover.losses import multiclass_dice_loss


class LossTest(unittest.TestCase):
    def test_perfect_logits_have_near_zero_dice_loss(self) -> None:
        target = torch.tensor([[[0, 1], [1, 0]]])
        logits = torch.full((1, 2, 2, 2), -20.0)
        logits.scatter_(1, target.unsqueeze(1), 20.0)
        loss = multiclass_dice_loss(
            logits, target, num_classes=2, ignore_index=255
        )
        self.assertLess(float(loss), 1e-6)

    def test_all_ignored_returns_differentiable_zero(self) -> None:
        target = torch.full((1, 2, 2), 255)
        logits = torch.randn(1, 2, 2, 2, requires_grad=True)
        loss = multiclass_dice_loss(
            logits, target, num_classes=2, ignore_index=255
        )
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
