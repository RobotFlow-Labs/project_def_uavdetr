"""Hungarian bipartite matcher for DETR-style detection.

Matches predicted queries to ground truth targets using optimal assignment
via scipy.optimize.linear_sum_assignment. Cost matrix combines classification,
L1 bbox, and GIoU costs.

Reference: Ultralytics RT-DETR (ultralytics/models/utils/ops.py)
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute GIoU between two sets of boxes in xyxy format.

    Returns [N, M] matrix of GIoU values.
    """
    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / (union + 1e-7)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - (enc_area - union) / (enc_area + 1e-7)
    return giou


class HungarianMatcher:
    """Bipartite matching between predictions and ground truth.

    Cost = cost_class * cls_cost + cost_bbox * l1_cost + cost_giou * giou_cost
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def __call__(
        self,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_boxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Match predictions to targets for each image in the batch.

        Args:
            pred_boxes: [B, num_queries, 4] in cxcywh normalised
            pred_logits: [B, num_queries, num_classes]
            gt_boxes: list of [Ni, 4] per image in cxcywh normalised
            gt_labels: list of [Ni] per image

        Returns:
            list of (pred_indices, gt_indices) tuples, one per image
        """
        bs, nq = pred_boxes.shape[:2]
        indices = []

        for i in range(bs):
            pb = pred_boxes[i]  # [nq, 4]
            pl = pred_logits[i]  # [nq, nc]
            gb = gt_boxes[i]  # [n_gt, 4]
            gl = gt_labels[i]  # [n_gt]

            n_gt = gb.shape[0]
            if n_gt == 0:
                indices.append(
                    (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
                )
                continue

            # Classification cost (focal loss formulation)
            pred_scores = pl.sigmoid()  # [nq, nc]
            # For each query-target pair, get the score for the target class
            pred_scores_for_gt = pred_scores[:, gl]  # [nq, n_gt]

            neg_cost = (
                (1 - self.focal_alpha)
                * pred_scores_for_gt**self.focal_gamma
                * (-(1 - pred_scores_for_gt + 1e-8).log())
            )
            pos_cost = (
                self.focal_alpha
                * (1 - pred_scores_for_gt) ** self.focal_gamma
                * (-(pred_scores_for_gt + 1e-8).log())
            )
            cost_class = pos_cost - neg_cost  # [nq, n_gt]

            # L1 bbox cost
            cost_bbox = torch.cdist(pb.float(), gb.float(), p=1)  # [nq, n_gt]

            # GIoU cost
            pb_xyxy = box_cxcywh_to_xyxy(pb)
            gb_xyxy = box_cxcywh_to_xyxy(gb)
            cost_giou = -generalized_box_iou(pb_xyxy, gb_xyxy)  # [nq, n_gt]

            # Final cost matrix
            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            # Hungarian assignment
            cost_np = cost.detach().cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(cost_np)
            indices.append(
                (
                    torch.tensor(pred_idx, dtype=torch.long),
                    torch.tensor(gt_idx, dtype=torch.long),
                )
            )

        return indices
