"""End-to-end model tests."""

from __future__ import annotations

import torch

from anima_def_uavdetr.model import DefUavDetr


@torch.inference_mode()
def test_model_emits_300_queries() -> None:
    model = DefUavDetr().eval()
    images = torch.randn(1, 3, 640, 640)

    boxes, logits = model(images)

    assert boxes.shape == (1, 300, 4)
    assert logits.shape == (1, 300, 1)


def test_model_returns_loss_dict_when_targets_present() -> None:
    model = DefUavDetr()
    images = torch.randn(1, 3, 640, 640)
    targets = {
        "boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32),
        "labels": torch.tensor([[0]], dtype=torch.long),
    }

    losses = model(images, targets=targets)

    assert "loss" in losses
    assert torch.isfinite(losses["loss"])
