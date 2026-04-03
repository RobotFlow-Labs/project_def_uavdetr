"""Prediction postprocessing for DEF-UAVDETR."""

from __future__ import annotations

import torch

from .losses import cxcywh_to_xyxy


def postprocess_queries(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    *,
    conf: float = 0.25,
    max_det: int = 300,
) -> list[torch.Tensor]:
    """Convert raw query outputs into `[x1, y1, x2, y2, score, class_id]` detections."""

    pred_xyxy = cxcywh_to_xyxy(pred_boxes).clamp_(0.0, 1.0)
    scores = pred_logits.sigmoid()
    detections: list[torch.Tensor] = []

    for boxes, logits in zip(pred_xyxy, scores, strict=True):
        class_scores, class_ids = logits.max(dim=-1)
        keep = class_scores >= conf
        boxes = boxes[keep]
        class_scores = class_scores[keep]
        class_ids = class_ids[keep]

        if boxes.numel() == 0:
            detections.append(torch.zeros((0, 6), dtype=pred_boxes.dtype, device=pred_boxes.device))
            continue

        order = class_scores.argsort(descending=True)[:max_det]
        selected = torch.cat(
            [
                boxes[order],
                class_scores[order, None],
                class_ids[order, None].to(boxes.dtype),
            ],
            dim=-1,
        )
        detections.append(selected)

    return detections
