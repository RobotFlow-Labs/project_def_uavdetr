"""Backbone shape tests."""

from __future__ import annotations

import torch

from anima_def_uavdetr.models.backbone import WTConvBackbone


@torch.inference_mode()
def test_backbone_feature_shapes() -> None:
    model = WTConvBackbone().eval()
    images = torch.randn(1, 3, 640, 640)
    s2, s3, s4, s5 = model(images)

    assert s2.shape == (1, 64, 160, 160)
    assert s3.shape == (1, 128, 80, 80)
    assert s4.shape == (1, 256, 40, 40)
    assert s5.shape == (1, 512, 20, 20)
