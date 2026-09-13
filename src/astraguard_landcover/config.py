"""YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    for section in ("data", "model", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing mapping '{section}' in {config_path}")
    return config


def with_runtime_overrides(
    config: dict[str, Any],
    *,
    data_dir: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    result = deepcopy(config)
    if data_dir:
        result["data"]["processed_dir"] = data_dir
    if output_dir:
        result["training"]["output_dir"] = output_dir
    return result

