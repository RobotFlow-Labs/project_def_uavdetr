"""Tests for DEF-UAVDETR settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from anima_def_uavdetr.config import UavDetrSettings, get_settings


def test_default_image_size() -> None:
    assert UavDetrSettings.model_fields["image_size"].default == 640


def test_settings_accept_explicit_paths(tmp_path: Path) -> None:
    settings = UavDetrSettings(
        uav_dataset_root=tmp_path / "uav",
        dut_dataset_root=tmp_path / "dut",
        checkpoints_root=tmp_path / "ckpt",
        export_root=tmp_path / "exports",
    )

    assert settings.uav_dataset_root == tmp_path / "uav"
    assert settings.dut_dataset_root == tmp_path / "dut"
    assert settings.best_checkpoint_path == tmp_path / "ckpt" / "uavdetr-best.pt"


def test_runtime_backend_validation_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="runtime_backend"):
        UavDetrSettings(runtime_backend="mlx")


def test_cached_settings_returns_same_instance() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
