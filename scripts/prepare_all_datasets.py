"""Prepare all UAV detection datasets into unified YOLO format.

Converts DUT-Anti-UAV (VOC XML) and DroneVehicle-night (DOTA rotated) to YOLO,
flattens BirdDrone nested dirs, and creates a unified manifest.

Usage:
    uv run python scripts/prepare_all_datasets.py

Output:
    /mnt/forge-data/shared_infra/datasets/uav_unified/
    ├── train/images/  (symlinks to all datasets)
    ├── train/labels/  (YOLO .txt files)
    ├── val/images/
    ├── val/labels/
    └── manifest.json
"""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
SERAPHIM = Path("/mnt/forge-data/datasets/uav_detection/seraphim")
BIRDDRONE = Path("/mnt/forge-data/shared_infra/datasets/lat_birddrone")
DRONEVEHICLE = Path("/mnt/forge-data/shared_infra/datasets/dronevehicle_night/DroneVehicle-night/rgb")
DUT = Path("/mnt/train-data/datasets/dut_anti_uav")
UNIFIED = Path("/mnt/forge-data/shared_infra/datasets/uav_unified")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── VOC XML → YOLO ────────────────────────────────────────────────────
def voc_xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert Pascal VOC XML to YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        x1 = float(bbox.find("xmin").text)
        y1 = float(bbox.find("ymin").text)
        x2 = float(bbox.find("xmax").text)
        y2 = float(bbox.find("ymax").text)
        # YOLO: class cx cy w h (normalised)
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ── DOTA rotated → YOLO axis-aligned ──────────────────────────────────
def dota_to_yolo(label_path: Path, img_w: int = 640, img_h: int = 512) -> list[str]:
    """Convert DOTA rotated bbox to YOLO axis-aligned bbox."""
    lines = []
    if not label_path.exists():
        return lines
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        coords = [float(p) for p in parts[:8]]
        cls_name = parts[8]
        # Only keep drone/UAV-related classes (class 0)
        # DroneVehicle has: car, truck, bus, van, freight_car — skip those
        # We want the drone itself but this dataset labels ground vehicles
        # Map all to class 0 for now (vehicle detection from drone perspective)
        xs = coords[0::2]
        ys = coords[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        if w > 0 and h > 0:
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


# ── main pipeline ─────────────────────────────────────────────────────
def main():
    for split in ("train", "val"):
        (UNIFIED / split / "images").mkdir(parents=True, exist_ok=True)
        (UNIFIED / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {}

    # ── 1. Seraphim (already YOLO) ─────────────────────────────────
    print("=== SERAPHIM ===")
    for split in ("train",):
        img_dir = SERAPHIM / split / "images"
        lbl_dir = SERAPHIM / split / "labels"
        n = 0
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS or not img.is_file():
                continue
            dst_img = UNIFIED / "train/images" / f"seraphim_{img.name}"
            dst_lbl = UNIFIED / "train/labels" / f"seraphim_{img.stem}.txt"
            if not dst_img.exists():
                os.symlink(img, dst_img)
            src_lbl = lbl_dir / f"{img.stem}.txt"
            if src_lbl.exists() and not dst_lbl.exists():
                os.symlink(src_lbl, dst_lbl)
            n += 1
        print(f"  {split}: {n} images")
        counts["seraphim"] = n

    # ── 2. BirdDrone (YOLO in nested dirs) ─────────────────────────
    print("=== BIRDDRONE ===")
    n_total = 0
    for category in ("drone2025/dronepicture", "bird2025/birdpicture"):
        img_base = BIRDDRONE / category / "images"
        lbl_base = BIRDDRONE / category / "labels"
        prefix = category.split("/")[0]
        # Nested: each subdir has frames
        for subdir in sorted(img_base.iterdir()):
            if subdir.is_dir():
                for img in subdir.iterdir():
                    if img.suffix.lower() not in IMG_EXTS:
                        continue
                    unique = f"birddrone_{prefix}_{subdir.name}_{img.name}"
                    dst_img = UNIFIED / "train/images" / unique
                    dst_lbl = UNIFIED / "train/labels" / f"birddrone_{prefix}_{subdir.name}_{img.stem}.txt"
                    if not dst_img.exists():
                        os.symlink(img, dst_img)
                    # Find matching label
                    lbl_subdir = lbl_base / subdir.name
                    src_lbl = lbl_subdir / f"{img.stem}.txt"
                    if src_lbl.exists() and not dst_lbl.exists():
                        os.symlink(src_lbl, dst_lbl)
                    n_total += 1
            elif subdir.suffix.lower() in IMG_EXTS:
                # Direct images (not nested)
                unique = f"birddrone_{prefix}_{subdir.name}"
                dst_img = UNIFIED / "train/images" / unique
                if not dst_img.exists():
                    os.symlink(subdir, dst_img)
                src_lbl = lbl_base / f"{subdir.stem}.txt"
                dst_lbl = UNIFIED / "train/labels" / f"birddrone_{prefix}_{subdir.stem}.txt"
                if src_lbl.exists() and not dst_lbl.exists():
                    os.symlink(src_lbl, dst_lbl)
                n_total += 1
    print(f"  total: {n_total} images")
    counts["birddrone"] = n_total

    # ── 3. DUT-Anti-UAV (VOC XML → YOLO) ──────────────────────────
    print("=== DUT-ANTI-UAV ===")
    for split_name, unified_split in [("train", "train"), ("val", "val")]:
        img_dir = DUT / split_name / "img"
        xml_dir = DUT / split_name / "xml"
        n = 0
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            xml_path = xml_dir / f"{img.stem}.xml"
            if not xml_path.exists():
                continue
            # Get image dimensions from XML
            tree = ET.parse(xml_path)
            size = tree.getroot().find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)
            # Convert
            yolo_lines = voc_xml_to_yolo(xml_path, img_w, img_h)
            # Symlink image
            dst_img = UNIFIED / f"{unified_split}/images/dut_{img.name}"
            if not dst_img.exists():
                os.symlink(img, dst_img)
            # Write YOLO label
            dst_lbl = UNIFIED / f"{unified_split}/labels/dut_{img.stem}.txt"
            if not dst_lbl.exists():
                dst_lbl.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")
            n += 1
        print(f"  {split_name}: {n} images")
        counts[f"dut_{split_name}"] = n

    # ── 4. DroneVehicle-night (DOTA → YOLO) ────────────────────────
    print("=== DRONEVEHICLE-NIGHT ===")
    for split_name, unified_split in [("train", "train"), ("val", "val")]:
        img_dir = DRONEVEHICLE / "image" / f"{split_name}_img"
        lbl_dir = DRONEVEHICLE / "label" / f"{split_name}_labelTxt"
        if not img_dir.exists():
            print(f"  {split_name}: dir not found, skipping")
            continue
        n = 0
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = lbl_dir / f"{img.stem}.txt"
            yolo_lines = dota_to_yolo(lbl_path)
            dst_img = UNIFIED / f"{unified_split}/images/dv_{img.name}"
            if not dst_img.exists():
                os.symlink(img, dst_img)
            dst_lbl = UNIFIED / f"{unified_split}/labels/dv_{img.stem}.txt"
            if not dst_lbl.exists():
                dst_lbl.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")
            n += 1
        print(f"  {split_name}: {n} images")
        counts[f"dronevehicle_{split_name}"] = n

    # ── Summary ────────────────────────────────────────────────────
    train_imgs = len(list((UNIFIED / "train/images").iterdir()))
    train_lbls = len(list((UNIFIED / "train/labels").iterdir()))
    val_imgs = len(list((UNIFIED / "val/images").iterdir()))
    val_lbls = len(list((UNIFIED / "val/labels").iterdir()))

    manifest = {
        "datasets": counts,
        "total_train": train_imgs,
        "total_val": val_imgs,
        "format": "YOLO (class cx cy w h normalised)",
        "image_exts": list(IMG_EXTS),
    }
    (UNIFIED / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== UNIFIED DATASET ===")
    print(f"Train: {train_imgs} images, {train_lbls} labels")
    print(f"Val:   {val_imgs} images, {val_lbls} labels")
    print(f"Total: {train_imgs + val_imgs}")
    print(f"Output: {UNIFIED}")
    print(f"Manifest: {UNIFIED / 'manifest.json'}")


if __name__ == "__main__":
    main()
