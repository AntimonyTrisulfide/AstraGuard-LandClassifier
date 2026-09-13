"""Streaming segmentation metrics derived from a confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .classes import AGRICULTURE_CLASS_ID, CLASS_NAMES


@dataclass
class SegmentationMetrics:
    num_classes: int
    ignore_index: int

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )

    @torch.no_grad()
    def update(self, logits_or_prediction: torch.Tensor, target: torch.Tensor) -> None:
        if logits_or_prediction.ndim == target.ndim + 1:
            prediction = logits_or_prediction.argmax(dim=1)
        else:
            prediction = logits_or_prediction
        prediction = prediction.detach().reshape(-1).to("cpu", torch.int64)
        target = target.detach().reshape(-1).to("cpu", torch.int64)
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        encoded = target[valid] * self.num_classes + prediction[valid]
        counts = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        matrix = self.confusion.numpy().astype(np.float64)
        true_positive = np.diag(matrix)
        target_count = matrix.sum(axis=1)
        predicted_count = matrix.sum(axis=0)
        union = target_count + predicted_count - true_positive
        precision = np.divide(
            true_positive,
            predicted_count,
            out=np.full_like(true_positive, np.nan),
            where=predicted_count > 0,
        )
        recall = np.divide(
            true_positive,
            target_count,
            out=np.full_like(true_positive, np.nan),
            where=target_count > 0,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.full_like(true_positive, np.nan),
            where=(precision + recall) > 0,
        )
        iou = np.divide(
            true_positive,
            union,
            out=np.full_like(true_positive, np.nan),
            where=union > 0,
        )
        total = matrix.sum()
        accuracy = float(true_positive.sum() / total) if total else float("nan")

        agriculture_true = target_count[AGRICULTURE_CLASS_ID]
        agriculture_predicted = predicted_count[AGRICULTURE_CLASS_ID]
        area_error = (
            float(abs(agriculture_predicted - agriculture_true) / agriculture_true * 100)
            if agriculture_true > 0
            else float("nan")
        )
        per_class = {}
        for index, class_name in enumerate(CLASS_NAMES):
            per_class[class_name] = {
                "precision": _json_float(precision[index]),
                "recall": _json_float(recall[index]),
                "f1": _json_float(f1[index]),
                "iou": _json_float(iou[index]),
                "support_pixels": int(target_count[index]),
                "predicted_pixels": int(predicted_count[index]),
            }
        return {
            "overall_accuracy": _json_float(accuracy),
            "macro_f1": _json_float(np.nanmean(f1)),
            "miou": _json_float(np.nanmean(iou)),
            "agriculture_area_error_percent": _json_float(area_error),
            "per_class": per_class,
            "confusion_matrix": self.confusion.tolist(),
        }


def _json_float(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None

