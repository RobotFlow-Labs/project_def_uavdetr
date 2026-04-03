"""Dataset layout discovery and validation for YOLO-format datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXPECTED_SPLITS = ("train", "valid", "test")


@dataclass
class DatasetLayout:
    """Describes a YOLO-format dataset directory structure."""

    root: Path
    split_names: tuple[str, ...] = EXPECTED_SPLITS

    def image_dir(self, split: str) -> Path:
        return self.root / split / "images"

    def label_dir(self, split: str) -> Path:
        return self.root / split / "labels"

    def validate(self) -> None:
        """Raise ``FileNotFoundError`` if any expected split is incomplete."""
        for split in self.split_names:
            img_dir = self.image_dir(split)
            if not img_dir.is_dir():
                raise FileNotFoundError(
                    f"missing image directory: {img_dir}"
                )


def build_dataset_layout(root: str | Path) -> DatasetLayout:
    """Build a ``DatasetLayout`` from a dataset root path.

    Always uses all expected splits so that ``validate()`` can detect
    missing directories.
    """
    return DatasetLayout(root=Path(root), split_names=EXPECTED_SPLITS)
