"""Typed settings and runtime defaults for DEF-UAVDETR."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_shared_root() -> Path:
    return Path("/Volumes/AIFlowDev/RobotFlowLabs/datasets")


class UavDetrSettings(BaseSettings):
    """Project settings for local build, training, and export workflows."""

    model_config = SettingsConfigDict(
        env_prefix="DEF_UAVDETR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    model_name: str = "def-uavdetr"
    package_name: str = "anima_def_uavdetr"
    image_size: int = 640
    frame_sample_stride: int = 5
    train_split_name: str = "train"
    val_split_name: str = "valid"
    test_split_name: str = "test"
    num_classes: int = 1
    class_names: tuple[str, ...] = ("uav",)
    runtime_backend: str = "auto"
    shared_data_root: Path = Field(default_factory=_default_shared_root)
    uav_dataset_root: Path = Field(
        default_factory=lambda: _default_shared_root() / "def_uavdetr" / "uav_dataset"
    )
    dut_dataset_root: Path = Field(
        default_factory=lambda: _default_shared_root() / "def_uavdetr" / "dut_anti_uav"
    )
    checkpoints_root: Path = Field(
        default_factory=lambda: _default_shared_root() / "models" / "def_uavdetr" / "checkpoints"
    )
    export_root: Path = Field(default_factory=lambda: Path("artifacts") / "exports")

    @field_validator("runtime_backend")
    @classmethod
    def validate_runtime_backend(cls, value: str) -> str:
        normalized = value.lower()
        allowed = {"auto", "cpu", "mps", "cuda"}
        if normalized not in allowed:
            msg = f"runtime_backend must be one of {sorted(allowed)}, got {value!r}"
            raise ValueError(msg)
        return normalized

    @field_validator(
        "shared_data_root",
        "uav_dataset_root",
        "dut_dataset_root",
        "checkpoints_root",
        "export_root",
        mode="before",
    )
    @classmethod
    def normalize_paths(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, value: int) -> int:
        if value <= 0 or value % 32 != 0:
            raise ValueError("image_size must be a positive multiple of 32")
        return value

    @field_validator("frame_sample_stride")
    @classmethod
    def validate_frame_sample_stride(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("frame_sample_stride must be positive")
        return value

    @computed_field
    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoints_root / "uavdetr-best.pt"

    @computed_field
    @property
    def onnx_export_path(self) -> Path:
        return self.export_root / "uavdetr.onnx"

    @computed_field
    @property
    def trt_fp16_export_path(self) -> Path:
        return self.export_root / "uavdetr.fp16.engine"


@lru_cache(maxsize=1)
def get_settings() -> UavDetrSettings:
    """Return a cached settings instance for application entry points."""

    return UavDetrSettings()
