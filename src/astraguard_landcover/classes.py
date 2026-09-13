"""The shared band and label contract used by every pipeline stage."""

from __future__ import annotations

from typing import Final

import numpy as np

BAND_NAMES: Final[tuple[str, ...]] = ("B02", "B03", "B04", "B08", "B11", "B12")
CLASS_NAMES: Final[tuple[str, ...]] = (
    "other",
    "agriculture",
    "built_up",
    "natural_vegetation",
    "water",
    "bare_or_sparse",
)
IGNORE_INDEX: Final[int] = 255
AGRICULTURE_CLASS_ID: Final[int] = 1

# ESA WorldCover 2021 v200 value -> model class id.
WORLDCOVER_TO_MODEL: Final[dict[int, int]] = {
    70: 0,
    90: 0,
    40: 1,
    50: 2,
    10: 3,
    20: 3,
    30: 3,
    95: 3,
    100: 3,
    80: 4,
    60: 5,
}

CLASS_COLORS: Final[dict[int, tuple[int, int, int, int]]] = {
    0: (180, 180, 180, 255),
    1: (255, 255, 100, 255),
    2: (250, 0, 0, 255),
    3: (0, 160, 0, 255),
    4: (0, 100, 255, 255),
    5: (190, 140, 90, 255),
    IGNORE_INDEX: (0, 0, 0, 0),
}


def remap_worldcover(mask: np.ndarray) -> np.ndarray:
    """Map raw WorldCover values to the fixed six-class task."""
    mapped = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for source_id, target_id in WORLDCOVER_TO_MODEL.items():
        mapped[mask == source_id] = target_id
    return mapped


def validate_class_contract(num_classes: int, in_channels: int) -> None:
    if num_classes != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output classes from the shared contract, "
            f"got {num_classes}."
        )
    if in_channels != len(BAND_NAMES):
        raise ValueError(
            f"Expected {len(BAND_NAMES)} input bands from the shared contract, "
            f"got {in_channels}."
        )

