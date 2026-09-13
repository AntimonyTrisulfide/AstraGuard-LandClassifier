import unittest
from types import SimpleNamespace

import torch
from torch import nn

from astraguard_landcover.models import _adapt_segformer_input


class SegFormerAdaptationTest(unittest.TestCase):
    def test_expands_rgb_projection_to_six_bands(self) -> None:
        projection = nn.Conv2d(3, 2, kernel_size=3, bias=True)
        with torch.no_grad():
            projection.weight[:, 0].fill_(1.0)
            projection.weight[:, 1].fill_(2.0)
            projection.weight[:, 2].fill_(3.0)
            projection.bias.fill_(4.0)

        model = SimpleNamespace(
            segformer=SimpleNamespace(
                encoder=SimpleNamespace(
                    patch_embeddings=[SimpleNamespace(proj=projection)]
                )
            ),
            config=SimpleNamespace(num_channels=3),
        )

        _adapt_segformer_input(model, 6)

        adapted = model.segformer.encoder.patch_embeddings[0].proj
        self.assertEqual(adapted.in_channels, 6)
        self.assertEqual(model.config.num_channels, 6)
        torch.testing.assert_close(adapted.weight[:, 0], torch.full_like(adapted.weight[:, 0], 1.5))
        torch.testing.assert_close(adapted.weight[:, 1], torch.full_like(adapted.weight[:, 1], 1.0))
        torch.testing.assert_close(adapted.weight[:, 2], torch.full_like(adapted.weight[:, 2], 0.5))
        torch.testing.assert_close(adapted.weight[:, 3], torch.full_like(adapted.weight[:, 3], 1.0))
        torch.testing.assert_close(adapted.bias, torch.full_like(adapted.bias, 4.0))


if __name__ == "__main__":
    unittest.main()
