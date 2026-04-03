"""Checkpoint tests."""

from __future__ import annotations

from pathlib import Path

import torch

from anima_def_uavdetr.checkpoints import load_checkpoint, save_checkpoint
from anima_def_uavdetr.model import DefUavDetr


def test_checkpoint_loader_accepts_plain_state_dict(tmp_path: Path) -> None:
    model = DefUavDetr()
    path = tmp_path / "plain.pt"
    torch.save(model.state_dict(), path)

    reloaded = DefUavDetr()
    load_checkpoint(reloaded, path)

    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key])


def test_checkpoint_saver_writes_nested_model_payload(tmp_path: Path) -> None:
    model = DefUavDetr()
    path = save_checkpoint(model, tmp_path / "nested.pt", metadata={"epoch": 1})
    payload = torch.load(path, map_location="cpu")

    assert "model" in payload
    assert payload["metadata"]["epoch"] == 1
