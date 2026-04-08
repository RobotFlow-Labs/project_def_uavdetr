"""DETR detection loss with Hungarian matching for DEF-UAVDETR.

Proper DETR loss: Hungarian matcher assigns each GT to exactly one query,
unmatched queries are trained as background (no object).

Paper-specific additions: Inner-CIoU + NWD hybrid bbox loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .matcher import HungarianMatcher

# ── Helper functions ───────────────────────────────────────────────────

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    half_w = w / 2
    half_h = h / 2
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def inner_ciou_loss(
    pred_boxes: torch.Tensor, target_boxes: torch.Tensor, *, inner_ratio: float = 0.8,
) -> torch.Tensor:
    """Inner-CIoU loss from the paper."""
    pred_inner = pred_boxes.clone()
    target_inner = target_boxes.clone()
    pred_inner[..., 2:] = pred_inner[..., 2:] * inner_ratio
    target_inner[..., 2:] = target_inner[..., 2:] * inner_ratio

    pred_xyxy = cxcywh_to_xyxy(pred_inner)
    target_xyxy = cxcywh_to_xyxy(target_inner)

    x1 = torch.maximum(pred_xyxy[..., 0], target_xyxy[..., 0])
    y1 = torch.maximum(pred_xyxy[..., 1], target_xyxy[..., 1])
    x2 = torch.minimum(pred_xyxy[..., 2], target_xyxy[..., 2])
    y2 = torch.minimum(pred_xyxy[..., 3], target_xyxy[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0) * (
        pred_xyxy[..., 3] - pred_xyxy[..., 1]
    ).clamp(min=0)
    area2 = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp(min=0) * (
        target_xyxy[..., 3] - target_xyxy[..., 1]
    ).clamp(min=0)
    iou = inter / (area1 + area2 - inter + 1e-7)

    center_dist = (pred_inner[..., :2] - target_inner[..., :2]).pow(2).sum(dim=-1)
    outer_x1 = torch.minimum(pred_xyxy[..., 0], target_xyxy[..., 0])
    outer_y1 = torch.minimum(pred_xyxy[..., 1], target_xyxy[..., 1])
    outer_x2 = torch.maximum(pred_xyxy[..., 2], target_xyxy[..., 2])
    outer_y2 = torch.maximum(pred_xyxy[..., 3], target_xyxy[..., 3])
    diag = (outer_x2 - outer_x1).pow(2) + (outer_y2 - outer_y1).pow(2) + 1e-7

    pw, ph = pred_inner[..., 2], pred_inner[..., 3]
    tw, th = target_inner[..., 2], target_inner[..., 3]
    v = (4 / torch.pi**2) * (
        torch.atan(tw / (th + 1e-7)) - torch.atan(pw / (ph + 1e-7))
    ).pow(2)
    alpha = v / (1 - iou + v + 1e-7)
    ciou = iou - (center_dist / diag) - alpha * v
    return 1 - ciou


def normalized_wasserstein_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    nwd_constant: float = 12.8,
) -> torch.Tensor:
    """NWD loss from the paper."""
    center_diff = (pred_boxes[..., :2] - target_boxes[..., :2]).pow(2).sum(dim=-1)
    size_diff = (pred_boxes[..., 2:] - target_boxes[..., 2:]).pow(2).sum(dim=-1) / 4
    wasserstein = torch.sqrt(center_diff + size_diff + 1e-7)
    similarity = torch.exp(-wasserstein / nwd_constant)
    return 1 - similarity


def giou_loss_paired(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """Pairwise GIoU loss for matched box pairs."""
    x1 = torch.max(pred_xyxy[..., 0], gt_xyxy[..., 0])
    y1 = torch.max(pred_xyxy[..., 1], gt_xyxy[..., 1])
    x2 = torch.min(pred_xyxy[..., 2], gt_xyxy[..., 2])
    y2 = torch.min(pred_xyxy[..., 3], gt_xyxy[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * (pred_xyxy[..., 3] - pred_xyxy[..., 1])
    area2 = (gt_xyxy[..., 2] - gt_xyxy[..., 0]) * (gt_xyxy[..., 3] - gt_xyxy[..., 1])
    union = area1 + area2 - inter + 1e-7
    iou = inter / union

    enc_x1 = torch.min(pred_xyxy[..., 0], gt_xyxy[..., 0])
    enc_y1 = torch.min(pred_xyxy[..., 1], gt_xyxy[..., 1])
    enc_x2 = torch.max(pred_xyxy[..., 2], gt_xyxy[..., 2])
    enc_y2 = torch.max(pred_xyxy[..., 3], gt_xyxy[..., 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-7

    return 1 - (iou - (enc_area - union) / enc_area)


# ── Main DETR Loss ─────────────────────────────────────────────────────

class DETRDetectionLoss(nn.Module):
    """DETR detection loss with Hungarian matching.

    - Classification: focal BCE on ALL queries (matched=target, unmatched=background)
    - Bbox regression: L1 + GIoU + Inner-CIoU + NWD on MATCHED queries only
    """

    def __init__(
        self,
        *,
        num_classes: int = 1,
        cls_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        ciou_alpha: float = 0.6,
        nwd_constant: float = 12.8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.ciou_alpha = ciou_alpha
        self.nwd_constant = nwd_constant
        self.matcher = HungarianMatcher()

    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        targets: dict[str, list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute DETR loss.

        Args:
            pred_boxes: [B, nq, 4] cxcywh normalised (sigmoid)
            pred_logits: [B, nq, nc]
            targets: {'boxes': list of [Ni,4], 'labels': list of [Ni]}
        """
        device = pred_boxes.device
        bs, nq = pred_boxes.shape[:2]

        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]

        # ── 1. Hungarian matching ──────────────────────────────────
        match_indices = self.matcher(
            pred_boxes.detach().float(),
            pred_logits.detach().float(),
            gt_boxes, gt_labels,
        )

        # ── 2. Classification loss (ALL queries) ──────────────────
        target_cls = torch.zeros(bs, nq, self.num_classes, device=device)
        for i, (pred_idx, gt_idx) in enumerate(match_indices):
            if pred_idx.numel() > 0:
                cls_ids = gt_labels[i][gt_idx].to(device).long()
                cls_ids = cls_ids.clamp(0, self.num_classes - 1)
                target_cls[i, pred_idx, cls_ids] = 1.0

        n_pos = max(sum(idx[0].numel() for idx in match_indices), 1)

        # Focal BCE
        p = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target_cls, reduction="none")
        focal_weight = target_cls * (1 - p) ** 2 + (1 - target_cls) * p**2
        loss_cls = (0.25 * focal_weight * ce).sum() / n_pos

        # ── 3. Bbox losses (ONLY matched) ─────────────────────────
        matched_pred = []
        matched_gt = []
        for i, (pred_idx, gt_idx) in enumerate(match_indices):
            if pred_idx.numel() > 0:
                matched_pred.append(pred_boxes[i, pred_idx])
                matched_gt.append(gt_boxes[i][gt_idx].to(device))

        if matched_pred:
            all_pred = torch.cat(matched_pred)
            all_gt = torch.cat(matched_gt)
            n_m = all_pred.shape[0]

            loss_l1 = F.l1_loss(all_pred, all_gt, reduction="sum") / max(n_m, 1)

            pred_xyxy = cxcywh_to_xyxy(all_pred)
            gt_xyxy = cxcywh_to_xyxy(all_gt)
            loss_giou = giou_loss_paired(pred_xyxy, gt_xyxy).sum() / max(n_m, 1)

            loss_ciou = inner_ciou_loss(all_pred, all_gt).mean()
            loss_nwd = normalized_wasserstein_loss(
                all_pred, all_gt, nwd_constant=self.nwd_constant,
            ).mean()
        else:
            zero = pred_boxes.sum() * 0.0
            loss_l1 = zero
            loss_giou = zero
            loss_ciou = zero
            loss_nwd = zero

        # ── 4. Total ──────────────────────────────────────────────
        loss_bbox = loss_l1 + self.ciou_alpha * loss_ciou + (1 - self.ciou_alpha) * loss_nwd
        total = (
            self.cls_weight * loss_cls
            + self.bbox_weight * loss_bbox
            + self.giou_weight * loss_giou
        )

        return {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_bbox_l1": loss_l1.detach(),
            "loss_giou": loss_giou.detach(),
            "loss_inner_ciou": loss_ciou.detach(),
            "loss_nwd": loss_nwd.detach(),
        }


# Backward-compatible alias
HybridInnerCiouNwdLoss = DETRDetectionLoss
