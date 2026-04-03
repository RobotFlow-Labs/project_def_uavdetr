"""Tests for dataset layout and frame sampling helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from anima_def_uavdetr.data.layout import build_dataset_layout
from anima_def_uavdetr.data.sampler import sample_every_n, sample_training_frames


def _make_split(root: Path, name: str) -> None:
    (root / name / "images").mkdir(parents=True)
    (root / name / "labels").mkdir(parents=True)


def test_layout_uses_expected_split_names(tmp_path: Path) -> None:
    for split in ("train", "valid", "test"):
        _make_split(tmp_path, split)

    layout = build_dataset_layout(tmp_path)

    assert layout.split_names == ("train", "valid", "test")
    layout.validate()


def test_layout_validation_fails_on_missing_split_dirs(tmp_path: Path) -> None:
    _make_split(tmp_path, "train")
    _make_split(tmp_path, "valid")

    layout = build_dataset_layout(tmp_path)

    with pytest.raises(FileNotFoundError, match="missing image directory"):
        layout.validate()


def test_sample_every_five() -> None:
    assert sample_every_n(list(range(10)), stride=5) == [0, 5]


def test_sample_training_frames_is_seeded_per_group() -> None:
    frames = [Path(f"frame_{index:02d}.jpg") for index in range(10)]

    sampled = sample_training_frames(frames, stride=5, seed=7)

    assert sampled == [Path("frame_02.jpg"), Path("frame_06.jpg")]


def test_sample_training_frames_rejects_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride must be positive"):
        sample_training_frames([], stride=0)
