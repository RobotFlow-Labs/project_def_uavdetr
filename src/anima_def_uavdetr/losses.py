"""Hybrid box loss for DEF-UAVDETR."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    half_w = w / 2
    half_h = h / 2
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.minimum(boxes1[..., 3], boxes2[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + 1e-7
    return inter / union


def inner_ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, *, inner_ratio: float = 0.8) -> torch.Tensor:
    """Approximate Inner-CIoU by shrinking both boxes before applying CIoU terms."""

    pred_inner = pred_boxes.clone()
    target_inner = target_boxes.clone()
    pred_inner[..., 2:] = pred_inner[..., 2:] * inner_ratio
    target_inner[..., 2:] = target_inner[..., 2:] * inner_ratio

    pred_xyxy = cxcywh_to_xyxy(pred_inner)
    target_xyxy = cxcywh_to_xyxy(target_inner)
    iou = box_iou(pred_xyxy, target_xyxy)

    pred_center = pred_inner[..., :2]
    target_center = target_inner[..., :2]
    center_distance = (pred_center - target_center).pow(2).sum(dim=-1)

    outer_x1 = torch.minimum(pred_xyxy[..., 0], target_xyxy[..., 0])
    outer_y1 = torch.minimum(pred_xyxy[..., 1], target_xyxy[..., 1])
    outer_x2 = torch.maximum(pred_xyxy[..., 2], target_xyxy[..., 2])
    outer_y2 = torch.maximum(pred_xyxy[..., 3], target_xyxy[..., 3])
    diagonal = (outer_x2 - outer_x1).pow(2) + (outer_y2 - outer_y1).pow(2) + 1e-7

    pred_w, pred_h = pred_inner[..., 2], pred_inner[..., 3]
    target_w, target_h = target_inner[..., 2], target_inner[..., 3]
    v = (4 / torch.pi**2) * (torch.atan(target_w / (target_h + 1e-7)) - torch.atan(pred_w / (pred_h + 1e-7))).pow(2)
    alpha = v / (1 - iou + v + 1e-7)
    ciou = iou - (center_distance / diagonal) - alpha * v
    return 1 - ciou


def normalized_wasserstein_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    nwd_constant: float = 12.8,
) -> torch.Tensor:
    """Compute an NWD-style loss between box Gaussian approximations."""

    center_diff = (pred_boxes[..., :2] - target_boxes[..., :2]).pow(2).sum(dim=-1)
    size_diff = (pred_boxes[..., 2:] - target_boxes[..., 2:]).pow(2).sum(dim=-1) / 4
    wasserstein = torch.sqrt(center_diff + size_diff + 1e-7)
    similarity = torch.exp(-wasserstein / nwd_constant)
    return 1 - similarity


def _as_batch_targets(targets: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if isinstance(targets, dict):
        boxes = targets["boxes"]
        labels = targets["labels"]
        if isinstance(boxes, list):
            return boxes, labels
        if boxes.ndim == 2:
            return [boxes], [labels]
        return [boxes[index] for index in range(boxes.shape[0])], [labels[index] for index in range(labels.shape[0])]
    raise TypeError("targets must be a dict with 'boxes' and 'labels'")


class HybridInnerCiouNwdLoss(nn.Module):
    """Hybrid loss with explicit Inner-CIoU and NWD weighting."""

    def __init__(
        self,
        *,
        alpha: float = 0.6,
        cls_weight: float = 1.0,
        bbox_weight: float = 5.0,
        nwd_constant: float = 12.8,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.nwd_constant = nwd_constant

    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_boxes_batch, target_labels_batch = _as_batch_targets(targets)
        device = pred_boxes.device
        loss_cls = torch.tensor(0.0, device=device)
        loss_l1 = torch.tensor(0.0, device=device)
        loss_ciou = torch.tensor(0.0, device=device)
        loss_nwd = torch.tensor(0.0, device=device)
        valid_batches = 0

        for batch_index, (target_boxes, target_labels) in enumerate(zip(target_boxes_batch, target_labels_batch, strict=True)):
            target_boxes = target_boxes.to(device)
            target_labels = target_labels.to(device).long().reshape(-1)
            if target_boxes.numel() == 0:
                continue

            count = min(target_boxes.shape[0], pred_boxes.shape[1])
            batch_pred_boxes = pred_boxes[batch_index, :count]
            batch_pred_logits = pred_logits[batch_index, :count]
            batch_target_boxes = target_boxes[:count]
            batch_target_labels = target_labels[:count]

            target_scores = torch.zeros_like(batch_pred_logits)
            target_scores.scatter_(1, batch_target_labels.unsqueeze(-1), 1.0)

            loss_cls = loss_cls + F.binary_cross_entropy_with_logits(batch_pred_logits, target_scores)
            loss_l1 = loss_l1 + F.l1_loss(batch_pred_boxes, batch_target_boxes)
            loss_ciou = loss_ciou + inner_ciou_loss(batch_pred_boxes, batch_target_boxes).mean()
            loss_nwd = loss_nwd + normalized_wasserstein_loss(
                batch_pred_boxes,
                batch_target_boxes,
                nwd_constant=self.nwd_constant,
            ).mean()
            valid_batches += 1

        if valid_batches == 0:
            total = pred_boxes.sum() * 0.0
            return {
                "loss": total,
                "loss_cls": total,
                "loss_bbox_l1": total,
                "loss_inner_ciou": total,
                "loss_nwd": total,
            }

        loss_cls = loss_cls / valid_batches
        loss_l1 = loss_l1 / valid_batches
        loss_ciou = loss_ciou / valid_batches
        loss_nwd = loss_nwd / valid_batches
        loss_bbox = loss_l1 + self.alpha * loss_ciou + (1 - self.alpha) * loss_nwd
        total = self.cls_weight * loss_cls + self.bbox_weight * loss_bbox
        return {
            "loss": total,
            "loss_cls": loss_cls,
            "loss_bbox_l1": loss_l1,
            "loss_inner_ciou": loss_ciou,
            "loss_nwd": loss_nwd,
        }
