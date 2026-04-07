"""Prepare unified_v2 dataset: ALL UAV datasets (~297K images).

Adds Baidu UAV and VisDrone to the existing unified dataset.
All classes remapped to 0 (single-class UAV/object detection).

Usage:
    uv run python scripts/prepare_unified_v2.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ── new datasets to add ───────────────────────────────────────────────
BAIDU = Path("/mnt/train-data/datasets/uav_dataset_baidu/uav_dataset")
VISDRONE = Path("/mnt/train-data/datasets/visdrone/yolo")
UNIFIED = Path("/mnt/forge-data/shared_infra/datasets/uav_unified")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def add_yolo_dataset(
    name: str, img_dir: Path, lbl_dir: Path, unified_split: str = "train",
) -> int:
    """Symlink a YOLO dataset into the unified dir. Returns count."""
    dst_img_dir = UNIFIED / unified_split / "images"
    dst_lbl_dir = UNIFIED / unified_split / "labels"
    n = 0
    for img in img_dir.iterdir():
        if img.suffix.lower() not in IMG_EXTS or not img.is_file():
            continue
        unique_img = f"{name}_{img.name}"
        unique_lbl = f"{name}_{img.stem}.txt"
        dst_img = dst_img_dir / unique_img
        dst_lbl = dst_lbl_dir / unique_lbl
        if not dst_img.exists():
            os.symlink(img.resolve(), dst_img)
        src_lbl = lbl_dir / f"{img.stem}.txt"
        if src_lbl.exists() and not dst_lbl.exists():
            os.symlink(src_lbl.resolve(), dst_lbl)
        n += 1
    return n


def main():
    counts = {}

    # ── Baidu UAV (already YOLO) ───────────────────────────────────
    print("=== BAIDU UAV ===")
    n = add_yolo_dataset(
        "baidu", BAIDU / "train/images", BAIDU / "train/labels", "train",
    )
    print(f"  train: {n} images")
    counts["baidu_train"] = n

    n = add_yolo_dataset(
        "baidu", BAIDU / "valid/images", BAIDU / "valid/labels", "val",
    )
    print(f"  val: {n} images")
    counts["baidu_val"] = n

    # ── VisDrone (YOLO, images/all + labels/all) ───────────────────
    print("=== VISDRONE ===")
    n = add_yolo_dataset(
        "visdrone", VISDRONE / "images/all", VISDRONE / "labels/all", "train",
    )
    print(f"  train: {n} images")
    counts["visdrone"] = n

    # ── Rebuild file list caches ───────────────────────────────────
    print("\n=== REBUILDING FILE CACHES ===")
    for split in ("train", "val"):
        img_dir = UNIFIED / split / "images"
        cache = UNIFIED / f"{split}_files.txt"
        files = sorted([
            str(p) for p in img_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS
        ])
        cache.write_text("\n".join(files) + "\n")
        print(f"  {split}: {len(files)} files cached")

    # ── Update manifest ────────────────────────────────────────────
    manifest_path = UNIFIED / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["datasets_v2"] = counts
    manifest["total_train_v2"] = len(
        list((UNIFIED / "train/images").iterdir())
    )
    manifest["total_val_v2"] = len(
        list((UNIFIED / "val/images").iterdir())
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n=== UNIFIED v2 COMPLETE ===")
    print(f"Train: {manifest['total_train_v2']}")
    print(f"Val: {manifest['total_val_v2']}")
    print(f"Total: {manifest['total_train_v2'] + manifest['total_val_v2']}")


if __name__ == "__main__":
    main()
