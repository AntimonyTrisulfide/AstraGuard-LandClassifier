"""Download pretrained weights once on a network-enabled login/transfer node."""

from __future__ import annotations

import argparse
from pathlib import Path

from astraguard_landcover.config import load_config
from astraguard_landcover.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+")
    args = parser.parse_args()
    for config_path in args.configs:
        config = load_config(config_path)
        print(f"Caching model assets for {config_path}")
        model = build_model(config["model"])
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"  ready: {parameter_count:,} parameters")


if __name__ == "__main__":
    main()

