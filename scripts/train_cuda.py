"""CUDA-optimized training pipeline for DEF-UAVDETR on Seraphim dataset.

Features:
- Memory-mapped tensor cache (zero CPU bottleneck)
- torch.compile() kernel fusion
- torch.amp mixed precision (fp16 forward, fp32 loss)
- Cosine LR with warmup
- Checkpoint manager (keep best 2)
- Early stopping (patience 20)
- NaN detection + gradient clipping

Usage:
    CUDA_VISIBLE_DEVICES=6 nohup uv run python scripts/train_cuda.py \\
        --epochs 100 --batch-size auto \\
        > /mnt/artifacts-datai/logs/project_def_uavdetr/train_$(date +%Y%m%d_%H%M).log 2>&1 &
    disown
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from anima_def_uavdetr.model import DefUavDetr

# ── paths ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("/mnt/forge-data/shared_infra/datasets")
CKPT_DIR = Path("/mnt/artifacts-datai/checkpoints/project_def_uavdetr")
LOG_DIR = Path("/mnt/artifacts-datai/logs/project_def_uavdetr")
TB_DIR = Path("/mnt/artifacts-datai/tensorboard/project_def_uavdetr")

for d in (CKPT_DIR, LOG_DIR, TB_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── dataset ────────────────────────────────────────────────────────────
class TensorCacheDataset(Dataset):
    """Load from pre-built tensor cache — zero PIL, zero CPU decode."""

    def __init__(self, split: str = "train"):
        img_path = CACHE_DIR / f"seraphim_{split}_images.pt"
        tgt_path = CACHE_DIR / f"seraphim_{split}_targets.pt"
        print(f"[DATA] Loading {img_path} ...")
        self.images: torch.Tensor = torch.load(img_path, weights_only=True)  # [N,3,640,640] uint8
        print(f"[DATA] Loading {tgt_path} ...")
        self.targets: list[dict] = torch.load(tgt_path, weights_only=False)  # noqa: S301
        print(f"[DATA] {split}: {len(self.images)} images loaded to RAM")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx].float() / 255.0  # normalize on-the-fly
        tgt = self.targets[idx]
        return img, tgt


def collate_fn(batch):
    """Collate variable-length targets — keep per-image structure."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def targets_to_device(targets: list[dict], device: torch.device) -> dict:
    """Convert per-image target list to loss-function-compatible dict.

    The loss function's ``_as_batch_targets`` expects either:
    - boxes as a list of per-image tensors, or
    - boxes as a 3D tensor [B, N, 4]

    We pass lists to handle variable target counts per image.
    """
    return {
        "boxes": [t["boxes"].to(device) for t in targets],
        "labels": [t["labels"].to(device) for t in targets],
    }


# ── model ──────────────────────────────────────────────────────────────
def build_model(num_classes: int = 1, compile_model: bool = True):
    """Build DefUavDetr and optionally torch.compile."""
    model = DefUavDetr(num_classes=num_classes, num_queries=300)
    params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] DefUavDetr: {params:,} parameters")

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[MODEL] torch.compile() enabled (reduce-overhead)")
        except Exception as e:
            print(f"[MODEL] torch.compile() failed: {e} — using eager mode")

    return model


# ── LR scheduler ──────────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]


# ── checkpoint manager ─────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, mode: str = "min"):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)

        # Copy best
        best_val, best_path = self.history[0]
        shutil.copy2(best_path, self.save_dir / "best.pth")
        return path


# ── early stopping ─────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = (
            (metric < self.best - self.min_delta)
            if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── batch size finder ──────────────────────────────────────────────────
def find_batch_size(model: nn.Module, device: torch.device, target_util: float = 0.65) -> int:
    """Binary search for max batch size at target VRAM utilisation.

    Uses training mode with a backward pass to get realistic memory usage.
    """
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    total_mem = torch.cuda.get_device_properties(device).total_mem
    target_bytes = int(total_mem * target_util)

    bs = 2
    best_bs = 2
    while bs <= 128:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            dummy = torch.randn(bs, 3, 640, 640, device=device, requires_grad=False)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred_boxes, pred_logits = model(dummy)
                # Simulate backward pass for realistic memory measurement
                fake_loss = pred_boxes.sum() + pred_logits.sum()
            fake_loss.backward()
            peak = torch.cuda.max_memory_allocated(device)
            util = peak / total_mem
            print(f"  bs={bs}: {peak / 1e9:.2f} GB ({util * 100:.1f}%)")
            model.zero_grad(set_to_none=True)
            if peak <= target_bytes:
                best_bs = bs
                bs *= 2
            else:
                break
        except (RuntimeError, torch.OutOfMemoryError):
            print(f"  bs={bs}: OOM")
            break
        finally:
            torch.cuda.empty_cache()

    return best_bs


# ── training loop ──────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CONFIG] device={device}, epochs={args.epochs}, lr={args.lr}")
    print(f"[CONFIG] checkpoint_dir={CKPT_DIR}")
    print(f"[CONFIG] log_dir={LOG_DIR}")

    # ── GPU info ───────────────────────────────────────────────────
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"[GPU] {props.name}, {props.total_mem / 1e9:.1f} GB VRAM")

    # ── model ──────────────────────────────────────────────────────
    model = build_model(num_classes=1, compile_model=args.compile)
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        raw_model = model._orig_mod
    else:
        raw_model = model
    model.to(device)

    # ── batch size ─────────────────────────────────────────────────
    if args.batch_size == 0:
        print("[BATCH] Auto-detecting optimal batch size (with backward pass)...")
        args.batch_size = find_batch_size(raw_model, device, target_util=0.65)
    print(f"[BATCH] batch_size={args.batch_size}")

    # ── data ───────────────────────────────────────────────────────
    train_ds = TensorCacheDataset("train")
    val_ds = TensorCacheDataset("val")
    print(f"[DATA] train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── optimizer + scheduler ──────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    # ── resume ─────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume and Path(args.resume).exists():
        print(f"[RESUME] Loading {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)  # noqa: S301
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        print(f"[RESUME] Continuing from epoch {start_epoch}, step {global_step}")

    # ── managers ───────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(CKPT_DIR, keep_top_k=2, mode="min")
    early_stop = EarlyStopping(patience=args.patience, min_delta=1e-4)

    # ── print config ───────────────────────────────────────────────
    print(f"[TRAIN] epochs={args.epochs}, lr={args.lr}, wd={args.weight_decay}")
    print(f"[TRAIN] warmup={warmup_steps} steps, total={total_steps} steps")
    print(f"[TRAIN] steps_per_epoch={len(train_loader)}")

    # ── check VRAM after first batch ───────────────────────────────
    vram_checked = False
    metrics_path = LOG_DIR / "metrics.jsonl"

    with open(metrics_path, "a") as metrics_file:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_steps = 0
            t_epoch = time.time()

            for _batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                gpu_targets = targets_to_device(targets, device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss_dict = model(images, gpu_targets)
                    if isinstance(loss_dict, dict):
                        loss = sum(loss_dict.values())
                    else:
                        loss = loss_dict

                # NaN detection
                if torch.isnan(loss):
                    print(f"[FATAL] NaN loss at epoch {epoch}, step {global_step}")
                    print("[FIX] Reduce lr by 10x or check data")
                    return

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                epoch_steps += 1
                global_step += 1

                # VRAM check after first batch
                if not vram_checked and device.type == "cuda":
                    used = torch.cuda.memory_allocated(device) / 1e9
                    total = torch.cuda.get_device_properties(device).total_mem / 1e9
                    pct = used / total * 100
                    print(f"[VRAM] {used:.1f}/{total:.1f} GB ({pct:.0f}%)")
                    if pct < 30:
                        print(f"[WARN] VRAM usage {pct:.0f}% — too low, consider larger batch")
                    vram_checked = True

                # Log every 50 steps
                if global_step % 50 == 0:
                    lr = scheduler.get_lr()
                    steps_per_sec = epoch_steps / (time.time() - t_epoch)
                    print(
                        f"[Step {global_step}] loss={loss_val:.4f} lr={lr:.2e} "
                        f"speed={steps_per_sec:.1f} steps/s"
                    )

            # ── validation ─────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    gpu_targets = targets_to_device(targets, device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        loss_dict = model(images, gpu_targets)
                        if isinstance(loss_dict, dict):
                            loss = sum(loss_dict.values())
                        else:
                            loss = loss_dict
                    val_loss += loss.item()
                    val_steps += 1

            avg_train = epoch_loss / max(epoch_steps, 1)
            avg_val = val_loss / max(val_steps, 1)
            epoch_time = time.time() - t_epoch
            lr = scheduler.get_lr()

            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
                f"lr={lr:.2e} time={epoch_time:.1f}s"
            )

            # Log metrics
            record = {
                "epoch": epoch + 1,
                "step": global_step,
                "train_loss": round(avg_train, 6),
                "val_loss": round(avg_val, 6),
                "lr": lr,
                "epoch_time_s": round(epoch_time, 1),
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

            # ── checkpoint ─────────────────────────────────────────
            state = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "config": vars(args),
            }
            ckpt_mgr.save(state, avg_val, global_step)

            # ── early stopping ─────────────────────────────────────
            if early_stop.step(avg_val):
                print(f"[EARLY STOP] No improvement for {args.patience} epochs")
                break

    print(f"[DONE] Training complete. Best val_loss={early_stop.best:.4f}")
    print(f"[DONE] Best checkpoint: {CKPT_DIR / 'best.pth'}")


def main():
    parser = argparse.ArgumentParser(description="DEF-UAVDETR CUDA training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto-detect")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()
