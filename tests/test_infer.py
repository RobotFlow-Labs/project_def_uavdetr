"""Inference tests."""

from __future__ import annotations

import torch

from anima_def_uavdetr.infer import DefUavDetrPredictor


@torch.inference_mode()
def test_predictor_returns_xyxy_score_class() -> None:
    predictor = DefUavDetrPredictor(device="cpu")
    images = torch.randn(1, 3, 320, 320)

    detections = predictor.predict(images, conf=0.0)

    assert len(detections) == 1
    assert detections[0].shape[1] == 6
