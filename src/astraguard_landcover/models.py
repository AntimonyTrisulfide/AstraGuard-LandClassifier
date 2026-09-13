"""Model factory for the two agreed baselines."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .classes import BAND_NAMES, CLASS_NAMES, validate_class_contract


class SegFormerAdapter(nn.Module):
    """Make Hugging Face SegFormer return input-resolution logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=inputs).logits
        return F.interpolate(
            logits,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def _adapt_segformer_input(model: nn.Module, in_channels: int) -> None:
    """Expand an RGB pretrained patch projection to the multispectral contract."""
    projection = model.segformer.encoder.patch_embeddings[0].proj
    if projection.in_channels == in_channels:
        return
    if projection.in_channels != 3:
        raise ValueError(
            f"Can only adapt a 3-channel pretrained projection, got "
            f"{projection.in_channels} channels"
        )
    replacement = nn.Conv2d(
        in_channels,
        projection.out_channels,
        kernel_size=projection.kernel_size,
        stride=projection.stride,
        padding=projection.padding,
        dilation=projection.dilation,
        groups=projection.groups,
        bias=projection.bias is not None,
    )
    with torch.no_grad():
        old = projection.weight
        replacement.weight.zero_()
        # Pretrained input is RGB; project contract begins blue, green, red.
        replacement.weight[:, 0] = old[:, 2]
        replacement.weight[:, 1] = old[:, 1]
        replacement.weight[:, 2] = old[:, 0]
        if in_channels > 3:
            mean_kernel = old.mean(dim=1)
            for channel in range(3, in_channels):
                replacement.weight[:, channel] = mean_kernel
        # Preserve roughly the same total activation scale after adding bands.
        replacement.weight.mul_(3.0 / float(in_channels))
        if projection.bias is not None:
            replacement.bias.copy_(projection.bias)
    model.segformer.encoder.patch_embeddings[0].proj = replacement
    model.config.num_channels = in_channels


def _build_segformer(
    config: dict[str, Any],
    num_classes: int,
    in_channels: int,
    *,
    initialize_pretrained: bool,
) -> nn.Module:
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise ImportError(
            "SegFormer requires the 'models' dependencies: "
            "pip install -e '.[models]'"
        ) from exc

    id2label = {index: name for index, name in enumerate(CLASS_NAMES)}
    label2id = {name: index for index, name in id2label.items()}
    architecture = str(config.get("architecture", "b0")).lower()
    if architecture != "b0":
        raise ValueError(
            "This baseline currently reconstructs SegFormer architecture 'b0' only. "
            "Add and checkpoint an explicit architecture preset before using another size."
        )
    if bool(config.get("pretrained", True)) and initialize_pretrained:
        checkpoint = str(
            config.get("checkpoint", "nvidia/segformer-b0-finetuned-ade-512-512")
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        _adapt_segformer_input(model, in_channels)
    else:
        hf_config = SegformerConfig(
            num_channels=in_channels,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
        )
        model = SegformerForSemanticSegmentation(hf_config)
    return SegFormerAdapter(model)


def build_model(
    config: dict[str, Any], *, initialize_pretrained: bool = True
) -> nn.Module:
    name = str(config["name"]).lower()
    in_channels = int(config.get("in_channels", len(BAND_NAMES)))
    num_classes = int(config.get("num_classes", len(CLASS_NAMES)))
    validate_class_contract(num_classes, in_channels)

    if name == "deeplabv3plus":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "DeepLabV3+ requires the 'models' dependencies: "
                "pip install -e '.[models]'"
            ) from exc
        return smp.DeepLabV3Plus(
            encoder_name=str(config.get("encoder_name", "resnet34")),
            encoder_weights=(
                config.get("encoder_weights", "imagenet") if initialize_pretrained else None
            ),
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
    if name == "segformer":
        return _build_segformer(
            config,
            num_classes,
            in_channels,
            initialize_pretrained=initialize_pretrained,
        )
    raise ValueError(f"Unsupported model '{name}'. Use deeplabv3plus or segformer.")
