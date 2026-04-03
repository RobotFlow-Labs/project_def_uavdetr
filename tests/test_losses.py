"""Loss tests."""

from __future__ import annotations

import torch

from anima_def_uavdetr.losses import HybridInnerCiouNwdLoss


def test_hybrid_loss_returns_expected_terms() -> None:
    loss_fn = HybridInnerCiouNwdLoss(alpha=0.6, nwd_constant=12.8)
    pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]], dtype=torch.float32)
    pred_logits = torch.tensor([[[2.0], [-1.0]]], dtype=torch.float32)
    targets = {
        "boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32),
        "labels": torch.tensor([[0]], dtype=torch.long),
    }

    losses = loss_fn(pred_boxes, pred_logits, targets)

    assert set(losses) == {"loss", "loss_cls", "loss_bbox_l1", "loss_inner_ciou", "loss_nwd"}
    assert torch.isfinite(losses["loss"])
