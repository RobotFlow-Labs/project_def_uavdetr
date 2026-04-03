"""Build memory-mapped tensor cache from Seraphim YOLO dataset.

Creates a single .pt file with all images pre-loaded as uint8 tensors
and a separate targets file with YOLO annotations converted to DETR format.

Output:
  /mnt/forge-data/shared_infra/datasets/seraphim_train_images.pt   — [N, 3, 640, 640] uint8
  /mnt/forge-data/shared_infra/datasets/seraphim_train_targets.pt  — list of dicts
  /mnt/forge-data/shared_infra/datasets/seraphim_val_images.pt     — held-out 5%
  /mnt/forge-data/shared_infra/datasets/seraphim_val_targets.pt

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/build_tensor_cache.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

DATASET_ROOT = Path("/mnt/forge-data/datasets/uav_detection/seraphim")
CACHE_DIR = Path("/mnt/forge-data/shared_infra/datasets")
IMG_SIZE = 640
VAL_RATIO = 0.05  # 5% val from train


def load_yolo_labels(label_path: Path) -> torch.Tensor:
    """Load YOLO label file → tensor [N, 5] (class_id, cx, cy, w, h)."""
    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)
    lines = label_path.read_text().strip().split("\n")
    if not lines or lines[0] == "":
        return torch.zeros((0, 5), dtype=torch.float32)
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append([float(x) for x in parts[:5]])
    if not boxes:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def yolo_to_detr_target(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert YOLO [cls, cx, cy, w, h] to DETR target dict."""
    if labels.numel() == 0:
        return {
            "labels": torch.zeros(0, dtype=torch.long),
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
        }
    return {
        "labels": labels[:, 0].long(),
        "boxes": labels[:, 1:],  # cx, cy, w, h — already normalised
    }


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train_img_dir = DATASET_ROOT / "train" / "images"
    train_lbl_dir = DATASET_ROOT / "train" / "labels"

    # Gather all image files (exclude zips)
    img_files = sorted([
        f for f in train_img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.is_file()
    ])
    print(f"Found {len(img_files)} training images")

    # Split into train/val
    n_val = max(1, int(len(img_files) * VAL_RATIO))
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(img_files))
    val_indices = set(indices[:n_val].tolist())
    train_indices = [i for i in range(len(img_files)) if i not in val_indices]

    for split_name, split_indices in [("train", train_indices), ("val", sorted(val_indices))]:
        print(f"\nBuilding {split_name} cache: {len(split_indices)} images")
        t0 = time.time()

        images = torch.zeros((len(split_indices), 3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
        targets = []

        for out_idx, src_idx in enumerate(split_indices):
            img_path = img_files[src_idx]
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  WARN: cannot read {img_path}")
                targets.append(yolo_to_detr_target(torch.zeros((0, 5))))
                continue

            # Already 640x640 but handle edge cases
            if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # BGR→RGB, HWC→CHW
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[out_idx] = torch.from_numpy(rgb).permute(2, 0, 1)

            # Load labels
            stem = img_path.stem
            lbl_path = train_lbl_dir / f"{stem}.txt"
            labels = load_yolo_labels(lbl_path)
            targets.append(yolo_to_detr_target(labels))

            if (out_idx + 1) % 10000 == 0:
                elapsed = time.time() - t0
                rate = (out_idx + 1) / elapsed
                print(f"  {out_idx + 1}/{len(split_indices)} ({rate:.0f} img/s)")

        # Save
        img_path_out = CACHE_DIR / f"seraphim_{split_name}_images.pt"
        tgt_path_out = CACHE_DIR / f"seraphim_{split_name}_targets.pt"

        print(f"  Saving images: {img_path_out} ({images.shape})")
        torch.save(images, img_path_out)

        print(f"  Saving targets: {tgt_path_out} ({len(targets)} entries)")
        torch.save(targets, tgt_path_out)

        elapsed = time.time() - t0
        size_gb = img_path_out.stat().st_size / 1e9
        print(f"  Done in {elapsed:.1f}s — {size_gb:.2f} GB")

    print("\nCache build complete!")


if __name__ == "__main__":
    main()
