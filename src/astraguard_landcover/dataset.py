"""HDF5-backed PyTorch dataset with worker-safe lazy file handles."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .classes import BAND_NAMES


def load_stats(processed_dir: str | Path) -> dict[str, Any]:
    stats_path = Path(processed_dir) / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run astraguard-prepare before training."
        )
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    if len(stats["band_mean"]) != len(BAND_NAMES):
        raise ValueError("stats.json does not match the six-band data contract")
    return stats


class H5LandCoverDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Read one split without holding an HDF5 handle across worker processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        band_mean: list[float],
        band_std: list[float],
        scale_divisor: float = 10000.0,
        augment: bool = False,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Missing dataset split: {self.path}")
        with h5py.File(self.path, "r") as handle:
            self.length = int(handle["images"].shape[0])
            self.band_count = int(handle["images"].shape[1])
        if self.band_count != len(BAND_NAMES):
            raise ValueError(
                f"{self.path} has {self.band_count} bands; expected {len(BAND_NAMES)}"
            )
        self.mean = np.asarray(band_mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(band_std, dtype=np.float32)[:, None, None]
        if np.any(self.std <= 0):
            raise ValueError("Every band standard deviation must be positive")
        self.scale_divisor = float(scale_divisor)
        self.augment = augment
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return self.length

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)
        return self._handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        handle = self._file()
        image = np.asarray(handle["images"][index], dtype=np.float32)
        mask = np.asarray(handle["masks"][index], dtype=np.int64)
        raw_key = handle["keys"][index]
        key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)

        image /= self.scale_divisor
        image = (image - self.mean) / self.std

        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=2)
                mask = np.flip(mask, axis=1)
            if random.random() < 0.5:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=0)
            rotations = random.randrange(4)
            if rotations:
                image = np.rot90(image, rotations, axes=(1, 2))
                mask = np.rot90(mask, rotations, axes=(0, 1))

        # flip/rot90 may create negative strides, so copy before torch conversion.
        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            key,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        if self._handle is not None:
            self._handle.close()

