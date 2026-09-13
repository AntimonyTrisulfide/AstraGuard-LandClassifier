"""Combined class-weighted cross-entropy and multiclass Dice loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=logits.dtype)
    valid_channels = valid.unsqueeze(1)
    probabilities = torch.softmax(logits, dim=1) * valid_channels
    one_hot = one_hot * valid_channels
    dimensions = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
    present = one_hot.sum(dim=dimensions) > 0
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    if not torch.any(present):
        return logits.sum() * 0.0
    return 1.0 - dice[present].mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_index: int,
        class_weights: torch.Tensor | None,
        ce_weight: float,
        dice_weight: float,
    ) -> None:
        super().__init__()
        if ce_weight < 0 or dice_weight < 0 or ce_weight + dice_weight <= 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )
        dice = multiclass_dice_loss(
            logits,
            target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        return self.ce_weight * ce + self.dice_weight * dice

