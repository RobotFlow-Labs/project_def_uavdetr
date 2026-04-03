"""Frame sampling utilities for UAV-DETR dataset preparation.

Paper reference: Section 3.1, Section 4.1 — the paper samples 1 image
from every 5 adjacent frames to reduce temporal redundancy.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def sample_every_n(items: Sequence[T], stride: int = 5) -> list[T]:
    """Return every *stride*-th item starting from index 0."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    return [items[i] for i in range(0, len(items), stride)]


def sample_training_frames(
    paths: Sequence[Path],
    stride: int = 5,
    seed: int = 42,
) -> list[Path]:
    """Sample one random frame from each group of *stride* consecutive frames.

    This implements the paper's frame sampling strategy (Section 3.1):
    from every group of ``stride`` adjacent frames, one is randomly selected.
    The random selection is seeded per-group for reproducibility.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")

    sampled: list[Path] = []
    for i, start in enumerate(range(0, len(paths), stride)):
        group = list(paths[start : start + stride])
        rng = random.Random(seed + i)
        sampled.append(rng.choice(group))
    return sampled
