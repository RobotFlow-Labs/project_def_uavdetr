"""Neck tests."""

from __future__ import annotations

import torch

from anima_def_uavdetr.models.neck import ECFRFN


@torch.inference_mode()
def test_neck_outputs_four_feature_levels() -> None:
    neck = ECFRFN().eval()
    s2 = torch.randn(1, 64, 160, 160)
    s3 = torch.randn(1, 128, 80, 80)
    s4 = torch.randn(1, 256, 40, 40)
    f5 = torch.randn(1, 256, 20, 20)

    p2, p3, p4, p5 = neck(s2, s3, s4, f5)

    assert p2.shape == (1, 256, 160, 160)
    assert p3.shape == (1, 256, 80, 80)
    assert p4.shape == (1, 256, 40, 40)
    assert p5.shape == (1, 256, 20, 20)
