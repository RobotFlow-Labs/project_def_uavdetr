"""Adversarial robustness evaluation for DEF-UAVDETR.

Implements FGSM, PGD, and adversarial patch attacks from scratch using
torch autograd. Measures detection drop rate, confidence shift, and mAP
degradation under each attack.

Usage:
    uv run python scripts/adversarial_attack.py \
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anima_def_uavdetr.checkpoints import load_checkpoint  # noqa: E402
from anima_def_uavdetr.model import DefUavDetr  # noqa: E402
from anima_def_uavdetr.postprocess import postprocess_queries  # noqa: E402

logger = logging.getLogger("adversarial_attack")

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FGSMConfig:
    enabled: bool = True
    epsilons: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.04, 0.08])


@dataclass
class PGDConfig:
    enabled: bool = True
    epsilons: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.04])
    alpha: float = 0.004
    steps: int = 10
    random_start: bool = True


@dataclass
class PatchConfig:
    enabled: bool = True
    patch_sizes: list[int] = field(default_factory=lambda: [32, 64, 96])
    num_patches: int = 1
    optimize_steps: int = 50
    learning_rate: float = 0.01
    target_location: str = "on_object"


@dataclass
class AttackResult:
    attack_name: str
    parameter: str
    parameter_value: float | int
    clean_detections: int = 0
    attacked_detections: int = 0
    detection_drop_rate: float = 0.0
    clean_avg_conf: float = 0.0
    attacked_avg_conf: float = 0.0
    confidence_shift: float = 0.0
    clean_map50: float = 0.0
    attacked_map50: float = 0.0
    map50_degradation: float = 0.0
    num_images: int = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_image(path: Path, image_size: int = 640) -> torch.Tensor:
    """Load and preprocess a single image to [C, H, W] float32 in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor


def load_yolo_labels(label_path: Path) -> torch.Tensor:
    """Load YOLO-format labels as [N, 5] tensor: [class_id, cx, cy, w, h]."""
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
        cls_id = 0  # single class — remap everything to 0
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append([cls_id, cx, cy, w, h])
    if not boxes:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def collect_image_label_pairs(
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
    logger.info("Collected %d image-label pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# IoU & mAP computation
# ---------------------------------------------------------------------------


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of xyxy boxes. [N, 4] x [M, 4] -> [N, M]."""
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using 101-point interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    recall_points = np.linspace(0.0, 1.0, 101)
    interp = np.zeros_like(recall_points)
    for i, r in enumerate(recall_points):
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            interp[i] = mpre[idx[0]]
    return float(interp.mean())


def compute_map50(
    all_detections: list[torch.Tensor],
    all_gt_labels: list[torch.Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """Compute mAP@50 across all images. Single class assumed (class 0)."""
    all_scores = []
    all_tp = []
    total_gt = 0

    for dets, gt_labels in zip(all_detections, all_gt_labels, strict=True):
        gt_boxes_cxcywh = gt_labels[:, 1:5] if gt_labels.numel() > 0 else torch.zeros((0, 4))
        gt_boxes = (
            cxcywh_to_xyxy(gt_boxes_cxcywh) if gt_boxes_cxcywh.numel() > 0 else gt_boxes_cxcywh
        )
        num_gt = len(gt_boxes)
        total_gt += num_gt

        if dets.numel() == 0:
            continue

        pred_boxes = dets[:, :4]  # xyxy in [0,1]
        pred_scores = dets[:, 4]

        if num_gt == 0:
            for s in pred_scores.cpu().numpy():
                all_scores.append(s)
                all_tp.append(0)
            continue

        ious = box_iou_matrix(pred_boxes.cpu(), gt_boxes.cpu())
        matched_gt = set()
        order = pred_scores.argsort(descending=True)

        for idx in order:
            s = pred_scores[idx].item()
            all_scores.append(s)
            iou_row = ious[idx]
            best_gt = iou_row.argmax().item()
            if iou_row[best_gt] >= iou_threshold and best_gt not in matched_gt:
                all_tp.append(1)
                matched_gt.add(best_gt)
            else:
                all_tp.append(0)

    if total_gt == 0:
        return 0.0

    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)
    order = np.argsort(-all_scores)
    all_tp = all_tp[order]
    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)
    recalls = cum_tp / (total_gt + 1e-7)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-7)
    return compute_ap(recalls, precisions)


# ---------------------------------------------------------------------------
# Attack loss (maximize detection confidence to compute useful gradients,
# then we negate to suppress detections)
# ---------------------------------------------------------------------------


def detection_loss(model: DefUavDetr, images: torch.Tensor) -> torch.Tensor:
    """Compute a differentiable detection loss for adversarial attacks.

    We maximize the sum of top-k sigmoid logits — adversarial perturbation
    should minimize this to suppress detections.
    """
    pred_boxes, pred_logits = model(images)
    # Maximize confidence of all queries -> gradient points toward higher conf
    # Attacker negates gradient to suppress detections
    scores = pred_logits.sigmoid()
    top_scores = scores.max(dim=-1).values  # [B, num_queries]
    # Use top-50 most confident queries
    topk = min(50, top_scores.shape[-1])
    top_vals, _ = top_scores.topk(topk, dim=-1)
    return top_vals.sum()


# ---------------------------------------------------------------------------
# FGSM attack
# ---------------------------------------------------------------------------


def fgsm_attack(
    model: DefUavDetr,
    images: torch.Tensor,
    epsilon: float,
    device: torch.device,
) -> torch.Tensor:
    """Fast Gradient Sign Method — single-step L-inf attack."""
    images_adv = images.clone().detach().to(device).requires_grad_(True)
    loss = detection_loss(model, images_adv)
    loss.backward()
    grad_sign = images_adv.grad.sign()
    # Subtract gradient to suppress detections
    perturbed = images_adv.detach() - epsilon * grad_sign
    return perturbed.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# PGD attack
# ---------------------------------------------------------------------------


def pgd_attack(
    model: DefUavDetr,
    images: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool,
    device: torch.device,
) -> torch.Tensor:
    """Projected Gradient Descent — iterative L-inf attack."""
    original = images.clone().detach().to(device)
    if random_start:
        perturbed = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
        perturbed = perturbed.clamp(0.0, 1.0)
    else:
        perturbed = original.clone()

    for _ in range(steps):
        perturbed = perturbed.detach().requires_grad_(True)
        loss = detection_loss(model, perturbed)
        loss.backward()
        grad_sign = perturbed.grad.sign()
        perturbed = perturbed.detach() - alpha * grad_sign
        # Project back to epsilon-ball around original
        delta = (perturbed - original).clamp(-epsilon, epsilon)
        perturbed = (original + delta).clamp(0.0, 1.0)

    return perturbed.detach()


# ---------------------------------------------------------------------------
# Adversarial patch attack
# ---------------------------------------------------------------------------


def _get_patch_location(
    gt_labels: torch.Tensor,
    patch_size: int,
    image_size: int,
    mode: str,
) -> tuple[int, int]:
    """Determine patch placement coordinates (top-left)."""
    if mode == "on_object" and gt_labels.numel() > 0:
        # Place on first GT box center
        cx = int(gt_labels[0, 1].item() * image_size)
        cy = int(gt_labels[0, 2].item() * image_size)
        x = max(0, min(cx - patch_size // 2, image_size - patch_size))
        y = max(0, min(cy - patch_size // 2, image_size - patch_size))
        return x, y
    elif mode == "center":
        c = (image_size - patch_size) // 2
        return c, c
    else:  # random
        x = torch.randint(0, max(1, image_size - patch_size), (1,)).item()
        y = torch.randint(0, max(1, image_size - patch_size), (1,)).item()
        return x, y


def patch_attack(
    model: DefUavDetr,
    images: torch.Tensor,
    gt_labels: torch.Tensor,
    patch_size: int,
    optimize_steps: int,
    lr: float,
    target_location: str,
    device: torch.device,
) -> torch.Tensor:
    """Optimize an adversarial patch placed on/near detected objects."""
    b, c, h, w = images.shape
    # Initialize patch with random noise
    patch = torch.rand(1, c, patch_size, patch_size, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([patch], lr=lr)

    px, py = _get_patch_location(gt_labels, patch_size, h, target_location)

    for _ in range(optimize_steps):
        optimizer.zero_grad()
        clamped_patch = patch.clamp(0.0, 1.0)
        patched = images.clone()
        patched[:, :, py : py + patch_size, px : px + patch_size] = clamped_patch
        loss = detection_loss(model, patched)
        # Minimize detection confidence
        (-loss).backward()
        optimizer.step()

    # Apply optimized patch
    with torch.no_grad():
        result = images.clone()
        result[:, :, py : py + patch_size, px : px + patch_size] = patch.data.clamp(0.0, 1.0)
    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def run_clean_inference(
    model: DefUavDetr,
    image: torch.Tensor,
    device: torch.device,
    conf: float = 0.25,
    max_det: int = 300,
) -> torch.Tensor:
    """Run inference without attack, return detections [N, 6]."""
    with torch.no_grad():
        batch = image.unsqueeze(0).to(device)
        pred_boxes, pred_logits = model(batch)
        dets = postprocess_queries(pred_boxes, pred_logits, conf=conf, max_det=max_det)
    return dets[0]


def count_detections(dets: torch.Tensor) -> int:
    return dets.shape[0] if dets.numel() > 0 else 0


def avg_confidence(dets: torch.Tensor) -> float:
    if dets.numel() == 0 or dets.shape[0] == 0:
        return 0.0
    return float(dets[:, 4].mean().item())


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_attack(
    model: DefUavDetr,
    pairs: list[tuple[Path, Path]],
    attack_fn,
    attack_name: str,
    param_name: str,
    param_value: float | int,
    device: torch.device,
    conf: float = 0.25,
    max_det: int = 300,
    image_size: int = 640,
) -> AttackResult:
    """Run attack on all images and compute aggregate metrics."""
    result = AttackResult(
        attack_name=attack_name,
        parameter=param_name,
        parameter_value=param_value,
    )

    all_clean_dets: list[torch.Tensor] = []
    all_attack_dets: list[torch.Tensor] = []
    all_gt: list[torch.Tensor] = []
    total_clean = 0
    total_attacked = 0
    sum_clean_conf = 0.0
    sum_attack_conf = 0.0
    n_images = 0

    for img_path, label_path in pairs:
        try:
            image = load_image(img_path, image_size).to(device)
            gt = load_yolo_labels(label_path)
        except Exception as e:
            logger.warning("Skipping %s: %s", img_path.name, e)
            continue

        # Clean inference
        clean_dets = run_clean_inference(model, image, device, conf, max_det)

        # Attacked inference
        model.train()  # Enable gradients through model
        attacked_image = attack_fn(image.unsqueeze(0), gt)
        model.eval()

        with torch.no_grad():
            pred_boxes, pred_logits = model(attacked_image)
            attack_dets = postprocess_queries(pred_boxes, pred_logits, conf=conf, max_det=max_det)[
                0
            ]

        all_clean_dets.append(clean_dets.cpu())
        all_attack_dets.append(attack_dets.cpu())
        all_gt.append(gt)

        nc = count_detections(clean_dets)
        na = count_detections(attack_dets)
        total_clean += nc
        total_attacked += na
        sum_clean_conf += avg_confidence(clean_dets) * max(nc, 1)
        sum_attack_conf += avg_confidence(attack_dets) * max(na, 1)
        n_images += 1

        if n_images % 50 == 0:
            logger.info(
                "[%s eps=%.4f] Processed %d/%d images",
                attack_name,
                param_value,
                n_images,
                len(pairs),
            )

    if n_images == 0:
        return result

    result.num_images = n_images
    result.clean_detections = total_clean
    result.attacked_detections = total_attacked
    result.detection_drop_rate = (
        (1.0 - total_attacked / max(total_clean, 1)) if total_clean > 0 else 0.0
    )
    result.clean_avg_conf = sum_clean_conf / max(total_clean, 1)
    result.attacked_avg_conf = sum_attack_conf / max(total_attacked, 1)
    result.confidence_shift = result.attacked_avg_conf - result.clean_avg_conf
    result.clean_map50 = compute_map50(all_clean_dets, all_gt)
    result.attacked_map50 = compute_map50(all_attack_dets, all_gt)
    result.map50_degradation = result.clean_map50 - result.attacked_map50

    return result


def run_all_attacks(
    model: DefUavDetr,
    pairs: list[tuple[Path, Path]],
    device: torch.device,
    fgsm_cfg: FGSMConfig,
    pgd_cfg: PGDConfig,
    patch_cfg: PatchConfig,
    conf: float = 0.25,
    max_det: int = 300,
    image_size: int = 640,
) -> list[AttackResult]:
    """Run all configured attacks and return results."""
    results: list[AttackResult] = []

    # --- FGSM ---
    if fgsm_cfg.enabled:
        for eps in fgsm_cfg.epsilons:
            logger.info("=== FGSM eps=%.4f ===", eps)

            def fgsm_fn(imgs: torch.Tensor, _gt: torch.Tensor, _eps=eps) -> torch.Tensor:
                return fgsm_attack(model, imgs, _eps, device)

            res = evaluate_attack(
                model,
                pairs,
                fgsm_fn,
                "FGSM",
                "epsilon",
                eps,
                device,
                conf,
                max_det,
                image_size,
            )
            results.append(res)
            logger.info(
                "FGSM eps=%.4f: drop=%.1f%% conf_shift=%.4f mAP50_deg=%.4f",
                eps,
                res.detection_drop_rate * 100,
                res.confidence_shift,
                res.map50_degradation,
            )

    # --- PGD ---
    if pgd_cfg.enabled:
        for eps in pgd_cfg.epsilons:
            logger.info("=== PGD eps=%.4f steps=%d ===", eps, pgd_cfg.steps)
            alpha = pgd_cfg.alpha if pgd_cfg.alpha > 0 else eps / pgd_cfg.steps * 2.5

            def pgd_fn(
                imgs: torch.Tensor,
                _gt: torch.Tensor,
                _eps=eps,
                _alpha=alpha,
            ) -> torch.Tensor:
                return pgd_attack(
                    model,
                    imgs,
                    _eps,
                    _alpha,
                    pgd_cfg.steps,
                    pgd_cfg.random_start,
                    device,
                )

            res = evaluate_attack(
                model,
                pairs,
                pgd_fn,
                "PGD",
                "epsilon",
                eps,
                device,
                conf,
                max_det,
                image_size,
            )
            results.append(res)
            logger.info(
                "PGD eps=%.4f: drop=%.1f%% conf_shift=%.4f mAP50_deg=%.4f",
                eps,
                res.detection_drop_rate * 100,
                res.confidence_shift,
                res.map50_degradation,
            )

    # --- Patch ---
    if patch_cfg.enabled:
        for ps in patch_cfg.patch_sizes:
            logger.info("=== Patch size=%d ===", ps)

            def patch_fn(
                imgs: torch.Tensor,
                gt: torch.Tensor,
                _ps=ps,
            ) -> torch.Tensor:
                return patch_attack(
                    model,
                    imgs,
                    gt,
                    _ps,
                    patch_cfg.optimize_steps,
                    patch_cfg.learning_rate,
                    patch_cfg.target_location,
                    device,
                )

            res = evaluate_attack(
                model,
                pairs,
                patch_fn,
                "Patch",
                "patch_size",
                ps,
                device,
                conf,
                max_det,
                image_size,
            )
            results.append(res)
            logger.info(
                "Patch size=%d: drop=%.1f%% conf_shift=%.4f mAP50_deg=%.4f",
                ps,
                res.detection_drop_rate * 100,
                res.confidence_shift,
                res.map50_degradation,
            )

    return results


# ---------------------------------------------------------------------------
# TOML config loader
# ---------------------------------------------------------------------------


def load_toml_config(path: Path) -> dict:
    """Load TOML config file."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    return tomllib.loads(path.read_text())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adversarial robustness evaluation for DEF-UAVDETR",
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
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config file (default: configs/attack_config.toml)",
    )
    p.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:X or cpu)")
    p.add_argument("--max-samples", type=int, default=500, help="Max images to evaluate (0=all)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--image-size", type=int, default=640, help="Input image size")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/artifacts-datai/reports/project_def_uavdetr"),
        help="Output directory for JSON report",
    )
    p.add_argument("--no-fgsm", action="store_true", help="Disable FGSM attack")
    p.add_argument("--no-pgd", action="store_true", help="Disable PGD attack")
    p.add_argument("--no-patch", action="store_true", help="Disable patch attack")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config from TOML if provided
    fgsm_cfg = FGSMConfig()
    pgd_cfg = PGDConfig()
    patch_cfg = PatchConfig()

    config_path = args.config
    if config_path is None:
        default_config = Path(__file__).resolve().parent.parent / "configs" / "attack_config.toml"
        if default_config.exists():
            config_path = default_config

    if config_path is not None and config_path.exists():
        logger.info("Loading config from %s", config_path)
        cfg = load_toml_config(config_path)
        if "attack" in cfg:
            if "fgsm" in cfg["attack"]:
                fgsm_cfg = FGSMConfig(**cfg["attack"]["fgsm"])
            if "pgd" in cfg["attack"]:
                pgd_cfg = PGDConfig(**cfg["attack"]["pgd"])
            if "patch" in cfg["attack"]:
                patch_cfg = PatchConfig(**cfg["attack"]["patch"])

    # CLI overrides
    if args.no_fgsm:
        fgsm_cfg.enabled = False
    if args.no_pgd:
        pgd_cfg.enabled = False
    if args.no_patch:
        patch_cfg.enabled = False

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

    # Setup device
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
    pairs = collect_image_label_pairs(args.images, args.labels, args.max_samples)
    if not pairs:
        logger.error("No image-label pairs found")
        sys.exit(1)

    # Run attacks
    t0 = time.time()
    results = run_all_attacks(
        model,
        pairs,
        device,
        fgsm_cfg,
        pgd_cfg,
        patch_cfg,
        conf=args.conf,
        max_det=300,
        image_size=args.image_size,
    )
    elapsed = time.time() - t0

    # Build report
    report = {
        "model": "DEF-UAVDETR",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "num_images": len(pairs),
        "image_size": args.image_size,
        "conf_threshold": args.conf,
        "total_time_seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "attacks": [asdict(r) for r in results],
        "summary": {},
    }

    # Summary per attack type
    for attack_type in ["FGSM", "PGD", "Patch"]:
        type_results = [r for r in results if r.attack_name == attack_type]
        if type_results:
            worst = max(type_results, key=lambda r: r.detection_drop_rate)
            report["summary"][attack_type] = {
                "worst_drop_rate": round(worst.detection_drop_rate * 100, 2),
                "worst_parameter": f"{worst.parameter}={worst.parameter_value}",
                "worst_map50_degradation": round(worst.map50_degradation * 100, 2),
            }

    # Save report
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / f"adversarial_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report saved to %s", report_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("ADVERSARIAL ROBUSTNESS REPORT — DEF-UAVDETR")
    print("=" * 80)
    print(f"{'Attack':<10} {'Param':<15} {'Drop%':>8} {'ConfShift':>10} {'mAP50 Deg':>10}")
    print("-" * 53)
    for r in results:
        print(
            f"{r.attack_name:<10} {r.parameter}={r.parameter_value:<8} "
            f"{r.detection_drop_rate * 100:>7.1f}% "
            f"{r.confidence_shift:>+9.4f} "
            f"{r.map50_degradation * 100:>9.2f}%"
        )
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s | Images: {len(pairs)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
