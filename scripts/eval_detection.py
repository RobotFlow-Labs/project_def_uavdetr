"""Detection evaluation on test set for DEF-UAVDETR.

Computes mAP50, mAP75, mAP50:95, precision, recall, F1 on YOLO-format
ground truth. Compares against paper targets:
  mAP50:95 = 62.56%, P = 96.82%, R = 94.93%

Usage:
    uv run python scripts/eval_detection.py \
        --checkpoint /mnt/artifacts-datai/checkpoints/project_def_uavdetr/best.pth \
        --images /mnt/forge-data/shared_infra/datasets/uav_unified/val/images \
        --labels /mnt/forge-data/shared_infra/datasets/uav_unified/val/labels \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anima_def_uavdetr.checkpoints import load_checkpoint  # noqa: E402
from anima_def_uavdetr.model import DefUavDetr  # noqa: E402
from anima_def_uavdetr.postprocess import postprocess_queries  # noqa: E402

logger = logging.getLogger("eval_detection")

# Paper target metrics
PAPER_TARGETS = {
    "mAP50_95": 62.56,
    "precision": 96.82,
    "recall": 94.93,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_image(path: Path, image_size: int = 640) -> torch.Tensor:
    """Load image to [C, H, W] float32 in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def load_yolo_labels(label_path: Path) -> torch.Tensor:
    """Load YOLO labels as [N, 5]: [class_id, cx, cy, w, h] normalized."""
    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return torch.zeros((0, 5), dtype=torch.float32)
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # Remap all classes to 0 (single class detection)
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append([0.0, cx, cy, w, h])
    if not boxes:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def collect_pairs(
    images_dir: Path,
    labels_dir: Path,
    max_samples: int = 0,
) -> list[tuple[Path, Path]]:
    """Collect matched image-label pairs."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    pairs = []
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        pairs.append((img_path, label_path))
    if max_samples > 0:
        pairs = pairs[:max_samples]
    return pairs


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    if boxes.numel() == 0:
        return boxes
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between [N, 4] and [M, 4] xyxy boxes -> [N, M]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0] if boxes1.numel() else 0, boxes2.shape[0] if boxes2.numel() else 0)
        )
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


# ---------------------------------------------------------------------------
# AP / mAP computation
# ---------------------------------------------------------------------------


def compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using COCO-style 101-point interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    recall_points = np.linspace(0.0, 1.0, 101)
    interp = np.zeros_like(recall_points)
    for i, r in enumerate(recall_points):
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            interp[i] = mpre[idx[0]]
    return float(interp.mean())


def compute_map_at_iou(
    all_detections: list[torch.Tensor],
    all_gt_labels: list[torch.Tensor],
    iou_threshold: float,
) -> tuple[float, float, float]:
    """Compute AP, precision, recall at a given IoU threshold.

    Returns:
        (ap, precision_at_best_f1, recall_at_best_f1)
    """
    all_scores: list[float] = []
    all_tp: list[int] = []
    total_gt = 0

    for dets, gt_labels in zip(all_detections, all_gt_labels, strict=True):
        gt_cxcywh = gt_labels[:, 1:5] if gt_labels.numel() > 0 else torch.zeros((0, 4))
        gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)
        num_gt = len(gt_xyxy)
        total_gt += num_gt

        if dets.numel() == 0:
            continue

        pred_boxes = dets[:, :4]
        pred_scores = dets[:, 4]

        if num_gt == 0:
            for s in pred_scores.numpy():
                all_scores.append(float(s))
                all_tp.append(0)
            continue

        ious = box_iou_matrix(pred_boxes, gt_xyxy)
        matched_gt: set[int] = set()
        order = pred_scores.argsort(descending=True)

        for idx in order:
            score = float(pred_scores[idx].item())
            all_scores.append(score)
            iou_row = ious[idx]
            best_gt = int(iou_row.argmax().item())
            if iou_row[best_gt] >= iou_threshold and best_gt not in matched_gt:
                all_tp.append(1)
                matched_gt.add(best_gt)
            else:
                all_tp.append(0)

    if total_gt == 0:
        return 0.0, 0.0, 0.0

    scores_arr = np.array(all_scores)
    tp_arr = np.array(all_tp)
    order = np.argsort(-scores_arr)
    tp_arr = tp_arr[order]

    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(1 - tp_arr)
    recalls = cum_tp / (total_gt + 1e-7)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-7)

    ap = compute_ap_101(recalls, precisions)

    # Find best F1 operating point
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-7)
    best_idx = np.argmax(f1_scores)
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])

    return ap, best_precision, best_recall


def compute_all_metrics(
    all_detections: list[torch.Tensor],
    all_gt_labels: list[torch.Tensor],
) -> dict[str, float]:
    """Compute mAP50, mAP75, mAP50:95, P, R, F1."""
    # mAP50
    ap50, p50, r50 = compute_map_at_iou(all_detections, all_gt_labels, 0.5)

    # mAP75
    ap75, _, _ = compute_map_at_iou(all_detections, all_gt_labels, 0.75)

    # mAP50:95 (average over 10 thresholds)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for t in iou_thresholds:
        ap_t, _, _ = compute_map_at_iou(all_detections, all_gt_labels, t)
        aps.append(ap_t)
    map50_95 = float(np.mean(aps))

    f1 = 2 * p50 * r50 / (p50 + r50 + 1e-7)

    return {
        "mAP50": round(ap50 * 100, 2),
        "mAP75": round(ap75 * 100, 2),
        "mAP50_95": round(map50_95 * 100, 2),
        "precision": round(p50 * 100, 2),
        "recall": round(r50 * 100, 2),
        "F1": round(f1 * 100, 2),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: DefUavDetr,
    pairs: list[tuple[Path, Path]],
    device: torch.device,
    conf: float = 0.25,
    max_det: int = 300,
    image_size: int = 640,
    batch_size: int = 8,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]:
    """Run inference on all images. Returns (detections, gt_labels, per_image_results)."""
    all_dets: list[torch.Tensor] = []
    all_gt: list[torch.Tensor] = []
    per_image: list[dict] = []

    model.eval()
    total = len(pairs)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_pairs = pairs[start:end]

        images = []
        gts = []
        paths = []
        for img_path, label_path in batch_pairs:
            try:
                img = load_image(img_path, image_size)
                gt = load_yolo_labels(label_path)
                images.append(img)
                gts.append(gt)
                paths.append(img_path)
            except Exception as e:
                logger.warning("Skipping %s: %s", img_path.name, e)

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred_boxes, pred_logits = model(batch_tensor)
            dets = postprocess_queries(pred_boxes, pred_logits, conf=conf, max_det=max_det)

        for _i, (det, gt, p) in enumerate(zip(dets, gts, paths, strict=True)):
            det_cpu = det.cpu()
            all_dets.append(det_cpu)
            all_gt.append(gt)

            num_dets = det_cpu.shape[0] if det_cpu.numel() > 0 else 0
            num_gt = gt.shape[0] if gt.numel() > 0 else 0
            avg_conf = float(det_cpu[:, 4].mean().item()) if num_dets > 0 else 0.0

            per_image.append(
                {
                    "image": p.name,
                    "num_detections": num_dets,
                    "num_gt": num_gt,
                    "avg_confidence": round(avg_conf, 4),
                }
            )

        processed = min(end, total)
        if processed % 200 == 0 or processed == total:
            logger.info("Inference: %d/%d images processed", processed, total)

    return all_dets, all_gt, per_image


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detection evaluation for DEF-UAVDETR",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/artifacts-datai/checkpoints/project_def_uavdetr/best.pth"),
        help="Path to model checkpoint",
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("/mnt/forge-data/shared_infra/datasets/uav_unified/val/images"),
        help="Directory of test images",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("/mnt/forge-data/shared_infra/datasets/uav_unified/val/labels"),
        help="Directory of YOLO-format labels",
    )
    p.add_argument("--device", type=str, default="cuda:0", help="Device")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    p.add_argument("--image-size", type=int, default=640, help="Input image size")
    p.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    p.add_argument("--max-samples", type=int, default=0, help="Max images (0=all)")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/artifacts-datai/reports/project_def_uavdetr"),
        help="Output directory",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate paths
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)
    if not args.images.is_dir():
        logger.error("Images directory not found: %s", args.images)
        sys.exit(1)
    if not args.labels.is_dir():
        logger.error("Labels directory not found: %s", args.labels)
        sys.exit(1)

    # Device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", args.checkpoint)

    # Load model
    model = DefUavDetr(num_classes=1, num_queries=300)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %.2fM parameters", param_count / 1e6)

    # Collect data
    pairs = collect_pairs(args.images, args.labels, args.max_samples)
    logger.info("Collected %d image-label pairs", len(pairs))
    if not pairs:
        logger.error("No image-label pairs found")
        sys.exit(1)

    # Run inference
    t0 = time.time()
    all_dets, all_gt, per_image = run_inference(
        model,
        pairs,
        device,
        conf=args.conf,
        max_det=args.max_det,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    inference_time = time.time() - t0

    # Compute metrics
    metrics = compute_all_metrics(all_dets, all_gt)

    # Per-IoU AP breakdown
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_per_iou = {}
    for t in iou_thresholds:
        ap_t, _, _ = compute_map_at_iou(all_dets, all_gt, t)
        ap_per_iou[f"AP@{t:.2f}"] = round(ap_t * 100, 2)

    # Build report
    total_gt = sum(gt.shape[0] for gt in all_gt if gt.numel() > 0)
    total_dets = sum(d.shape[0] for d in all_dets if d.numel() > 0)

    report = {
        "model": "DEF-UAVDETR",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "image_size": args.image_size,
            "conf_threshold": args.conf,
            "max_det": args.max_det,
            "batch_size": args.batch_size,
        },
        "dataset": {
            "images_dir": str(args.images),
            "labels_dir": str(args.labels),
            "num_images": len(pairs),
            "total_gt_objects": total_gt,
            "total_detections": total_dets,
        },
        "metrics": metrics,
        "ap_per_iou": ap_per_iou,
        "paper_targets": PAPER_TARGETS,
        "paper_comparison": {
            "mAP50_95_delta": round(metrics["mAP50_95"] - PAPER_TARGETS["mAP50_95"], 2),
            "precision_delta": round(metrics["precision"] - PAPER_TARGETS["precision"], 2),
            "recall_delta": round(metrics["recall"] - PAPER_TARGETS["recall"], 2),
        },
        "timing": {
            "total_inference_seconds": round(inference_time, 2),
            "images_per_second": round(len(pairs) / inference_time, 2) if inference_time > 0 else 0,
        },
        "per_image_results": per_image,
    }

    # Save report
    args.output.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = args.output / f"eval_detection_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", report_path)

    # Print summary
    print("\n" + "=" * 72)
    print("DETECTION EVALUATION REPORT -- DEF-UAVDETR")
    print("=" * 72)
    print(f"  Images:     {len(pairs)}")
    print(f"  GT objects: {total_gt}")
    print(f"  Detections: {total_dets}")
    print(f"  Throughput: {len(pairs) / max(inference_time, 1e-6):.1f} img/s")
    print()
    print(f"  {'Metric':<12} {'Ours':>8} {'Paper':>8} {'Delta':>8}")
    print(f"  {'-' * 40}")
    print(
        f"  {'mAP50:95':<12} {metrics['mAP50_95']:>7.2f}% "
        f"{PAPER_TARGETS['mAP50_95']:>7.2f}% "
        f"{metrics['mAP50_95'] - PAPER_TARGETS['mAP50_95']:>+7.2f}%"
    )
    print(f"  {'mAP50':<12} {metrics['mAP50']:>7.2f}%")
    print(f"  {'mAP75':<12} {metrics['mAP75']:>7.2f}%")
    print(
        f"  {'Precision':<12} {metrics['precision']:>7.2f}% "
        f"{PAPER_TARGETS['precision']:>7.2f}% "
        f"{metrics['precision'] - PAPER_TARGETS['precision']:>+7.2f}%"
    )
    print(
        f"  {'Recall':<12} {metrics['recall']:>7.2f}% "
        f"{PAPER_TARGETS['recall']:>7.2f}% "
        f"{metrics['recall'] - PAPER_TARGETS['recall']:>+7.2f}%"
    )
    print(f"  {'F1':<12} {metrics['F1']:>7.2f}%")
    print()

    # AP per IoU breakdown
    print(f"  {'IoU Threshold':<15} {'AP':>8}")
    print(f"  {'-' * 25}")
    for iou_key, ap_val in ap_per_iou.items():
        print(f"  {iou_key:<15} {ap_val:>7.2f}%")
    print("=" * 72)
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
