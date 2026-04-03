"""Encoder tests."""

from __future__ import annotations

import torch

from anima_def_uavdetr.models.encoder import SWSAIFIEncoder


@torch.inference_mode()
def test_encoder_preserves_shape() -> None:
    encoder = SWSAIFIEncoder(dim=256).eval()
    x = torch.randn(2, 256, 20, 20)
    assert encoder(x).shape == x.shape
