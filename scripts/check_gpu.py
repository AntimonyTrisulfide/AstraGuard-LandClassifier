"""Fail unless the configured segmentation model can train on the allocated GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from astraguard_landcover.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA is unavailable to PyTorch {torch.__version__} "
            f"(wheel CUDA: {torch.version.cuda}). Refusing to train on CPU."
        )

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_config = config["model"]
    channels = int(model_config["in_channels"])
    model = build_model(model_config, initialize_pretrained=False).cuda().train()
    inputs = torch.randn(1, channels, 256, 256, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(inputs)
        loss = logits.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "wheel_cuda": torch.version.cuda,
                "cuda_available": True,
                "gpu_count": torch.cuda.device_count(),
                "device": torch.cuda.get_device_name(0),
                "model": model_config["name"],
                "amp_forward_backward": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
