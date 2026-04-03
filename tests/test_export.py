"""Export tests."""

from __future__ import annotations

from pathlib import Path

import torch

from anima_def_uavdetr.model import DefUavDetr
from scripts.export import SUPPORTED_EXPORTS, export_pytorch


def test_export_registry_contains_onnx() -> None:
    assert "onnx" in SUPPORTED_EXPORTS


def test_export_pytorch_writes_checkpoint(tmp_path: Path) -> None:
    model = DefUavDetr()
    output = export_pytorch(model, tmp_path / "model.pt")
    payload = torch.load(output, map_location="cpu")

    assert output.exists()
    assert "model" in payload
