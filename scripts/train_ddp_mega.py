"""8-GPU DDP training for Nighthawk Mega dataset (1.78M images).

Optimized for maximum throughput on 8x L4 (184GB total VRAM).
Supports multi-condition data (night, thermal, fog, rain, dusk, original).

Launch:
    PYTHONUNBUFFERED=1 PYTHONPATH="" \
    nohup .venv/bin/torchrun --nproc_per_node=8 --master_port=29600 \
        scripts/train_ddp_mega.py \
        --epochs 20 --batch-size 16 --lr 5e-6 \
        --resume /mnt/artifacts-datai/checkpoints/project_def_uavdetr/best.pth \
        > /mnt/artifacts-datai/logs/project_def_uavdetr/train_mega_$(date +%Y%m%d_%H%M).log 2>&1 &
    disown
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from anima_def_uavdetr.model import DefUavDetr

# ── paths ──────────────────────────────────────────────────────────────
MEGA_ROOT = Path("/mnt/forge-data/datasets/nighthawk_mega")
CKPT_DIR = Path("/mnt/artifacts-datai/checkpoints/project_def_uavdetr")
LOG_DIR = Path("/mnt/artifacts-datai/logs/project_def_uavdetr")

IMG_SIZE = 640
CONDITIONS = ("night", "thermal", "fog", "rain", "dusk", "original")


# ── dataset ────────────────────────────────────────────────────────────
class NighthawkMegaDataset(Dataset):
    """Load from Nighthawk Mega — multi-condition YOLO labels.

    Structure:
        nighthawk_mega/
        ├── images/{condition}/{dataset}/*.jpg
        └── labels/{condition}/{dataset}/yolo/*.txt
    """

    def __init__(
        self, root: Path, split: str = "train",
        conditions: tuple[str, ...] = CONDITIONS,
        img_size: int = 640, augment: bool = False,
        file_list: Path | None = None,
    ):
        self.root = root
        self.img_size = img_size
        self.augment = augment

        if file_list and file_list.exists():
            self.samples = [
                line.strip().split("\t")
                for line in file_list.read_text().strip().split("\n")
                if line.strip()
            ]
        else:
            # Discover all image/label pairs
            self.samples = []
            img_root = root / "images"
            lbl_root = root / "labels"
            for condition in conditions:
                cond_img = img_root / condition
                cond_lbl = lbl_root / condition
                if not cond_img.exists():
                    continue
                for dataset_dir in sorted(cond_img.iterdir()):
                    if not dataset_dir.is_dir():
                        continue
                    yolo_dir = cond_lbl / dataset_dir.name / "yolo"
                    for img in dataset_dir.iterdir():
                        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                            lbl = yolo_dir / f"{img.stem}.txt"
                            self.samples.append((str(img), str(lbl)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        elif img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                tensor = tensor.flip(-1)

        # Load YOLO labels
        lbl_path = Path(lbl_path)
        if lbl_path.exists():
            try:
                data = np.loadtxt(str(lbl_path), dtype=np.float32).reshape(-1, 5)
                target = {
                    "labels": torch.zeros(data.shape[0], dtype=torch.long),
                    "boxes": torch.from_numpy(data[:, 1:]),
                }
            except (ValueError, IndexError):
                target = {
                    "labels": torch.zeros(0, dtype=torch.long),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                }
        else:
            target = {
                "labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }
        return tensor, target


# ── Also support unified dataset as fallback ──────────────────────────
class UnifiedFileListDataset(Dataset):
    """Fast dataset from pre-cached file list (unified or unified_v2)."""

    def __init__(
        self, file_list: Path, label_dir: Path,
        img_size: int = 640, augment: bool = False,
    ):
        self.image_paths = [
            Path(p.strip()) for p in file_list.read_text().strip().split("\n") if p.strip()
        ]
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        elif img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)

        if self.augment and torch.rand(1).item() > 0.5:
            tensor = tensor.flip(-1)

        lbl_path = self.label_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            try:
                data = np.loadtxt(str(lbl_path), dtype=np.float32).reshape(-1, 5)
                target = {
                    "labels": torch.zeros(data.shape[0], dtype=torch.long),
                    "boxes": torch.from_numpy(data[:, 1:]),
                }
            except (ValueError, IndexError):
                target = {
                    "labels": torch.zeros(0, dtype=torch.long),
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                }
        else:
            target = {
                "labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }
        return tensor, target


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def targets_to_device(targets: list[dict], device: torch.device) -> dict:
    return {
        "boxes": [t["boxes"].to(device) for t in targets],
        "labels": [t["labels"].to(device) for t in targets],
    }


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
                self.total_steps - self.warmup_steps, 1,
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


# ── checkpoint + early stopping ────────────────────────────────────────
class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, mode: str = "min"):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int) -> Path:
        path = self.save_dir / f"mega_checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
        best_val, best_path = self.history[0]
        shutil.copy2(best_path, self.save_dir / "best.pth")
        return path


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4, mode: str = "min"):
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


# ── 8-GPU DDP training loop ───────────────────────────────────────────
def train(args):
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=15))
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = local_rank == 0

    if is_main:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[MEGA] world_size={world_size}, device={device}")
        print(f"[MEGA] effective_batch_size={args.batch_size * world_size}")
        props = torch.cuda.get_device_properties(device)
        print(f"[GPU] {props.name}, {props.total_memory / 1e9:.1f} GB VRAM")

    # ── model ──────────────────────────────────────────────────────
    model = DefUavDetr(num_classes=1, num_queries=300).to(device)
    if is_main:
        params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] DefUavDetr: {params:,} parameters")

    # ── load checkpoint ────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        if is_main:
            print(f"[RESUME] Loading {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)  # noqa: S301
        model.load_state_dict(ckpt["model"])
        if is_main:
            print(f"[RESUME] epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── data ───────────────────────────────────────────────────────
    if args.source == "mega" and MEGA_ROOT.exists():
        if is_main:
            print(f"[DATA] Source: Nighthawk Mega ({MEGA_ROOT})")
        file_list = MEGA_ROOT / "train_files.txt"
        train_ds = NighthawkMegaDataset(
            MEGA_ROOT, split="train", augment=True,
            file_list=file_list if file_list.exists() else None,
        )
        val_file_list = MEGA_ROOT / "val_files.txt"
        val_ds = NighthawkMegaDataset(
            MEGA_ROOT, split="val", augment=False,
            file_list=val_file_list if val_file_list.exists() else None,
        )
    else:
        # Fallback to unified_v2
        unified = Path("/mnt/forge-data/shared_infra/datasets/uav_unified")
        if is_main:
            print(f"[DATA] Source: unified_v2 ({unified})")
        train_ds = UnifiedFileListDataset(
            unified / "train_files.txt", unified / "train/labels",
            augment=True,
        )
        val_ds = UnifiedFileListDataset(
            unified / "val_files.txt", unified / "val/labels",
            augment=False,
        )

    if is_main:
        print(f"[DATA] train={len(train_ds)}, val={len(val_ds)}")

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=local_rank, shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=local_rank, shuffle=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, collate_fn=collate_fn, drop_last=True,
        pin_memory=False, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=min(args.workers, 2), collate_fn=collate_fn,
        pin_memory=False,
    )

    # ── optimizer ──────────────────────────────────────────────────
    scaled_lr = args.lr * world_size
    if is_main:
        print(f"[OPT] base_lr={args.lr}, scaled_lr={scaled_lr} (x{world_size})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.03)  # 3% warmup for mega
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    ckpt_mgr = CheckpointManager(CKPT_DIR, keep_top_k=2, mode="min") if is_main else None
    early_stop = EarlyStopping(patience=args.patience, min_delta=1e-4)

    if is_main:
        print(f"[TRAIN] epochs={args.epochs}, warmup={warmup_steps}, total_steps={total_steps}")
        print(f"[TRAIN] steps/epoch={len(train_loader)}")

    global_step = 0
    metrics_path = LOG_DIR / "metrics_mega.jsonl"

    with open(metrics_path, "a") if is_main else open("/dev/null", "a") as mf:
        for epoch in range(args.epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_steps = 0
            t_epoch = time.time()

            for _i, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                gpu_targets = targets_to_device(targets, device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss_dict = model(images, gpu_targets)
                    loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict

                if torch.isnan(loss):
                    if is_main:
                        print(f"[FATAL] NaN at epoch {epoch}, step {global_step}")
                    dist.destroy_process_group()
                    return

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                if is_main and global_step % 100 == 0:
                    lr = scheduler.get_lr()
                    sps = epoch_steps / (time.time() - t_epoch)
                    print(
                        f"[Step {global_step}] loss={loss.item():.4f} lr={lr:.2e} "
                        f"speed={sps:.1f} steps/s ({sps * world_size:.1f} eff)",
                    )

            # ── validation ─────────────────────────────────────────
            model.eval()
            val_loss_local = torch.tensor(0.0, device=device)
            val_steps_local = torch.tensor(0, device=device)
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    gpu_targets = targets_to_device(targets, device)
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        ld = model(images, gpu_targets)
                        loss = sum(ld.values()) if isinstance(ld, dict) else ld
                    val_loss_local += loss
                    val_steps_local += 1

            dist.all_reduce(val_loss_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_steps_local, op=dist.ReduceOp.SUM)

            avg_train = epoch_loss / max(epoch_steps, 1)
            avg_val = (val_loss_local / max(val_steps_local, 1)).item()
            epoch_time = time.time() - t_epoch

            if is_main:
                lr = scheduler.get_lr()
                print(
                    f"[Epoch {epoch + 1}/{args.epochs}] "
                    f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
                    f"lr={lr:.2e} time={epoch_time:.1f}s",
                )
                mf.write(json.dumps({
                    "epoch": epoch + 1, "step": global_step,
                    "train_loss": round(avg_train, 6), "val_loss": round(avg_val, 6),
                    "lr": lr, "epoch_time_s": round(epoch_time, 1),
                    "world_size": world_size,
                }) + "\n")
                mf.flush()

                state = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch, "step": global_step,
                    "train_loss": avg_train, "val_loss": avg_val,
                    "config": vars(args),
                }
                ckpt_mgr.save(state, avg_val, global_step)

            should_stop = torch.tensor(0, device=device)
            if is_main and early_stop.step(avg_val):
                print(f"[EARLY STOP] No improvement for {args.patience} epochs")
                should_stop = torch.tensor(1, device=device)
            dist.broadcast(should_stop, src=0)
            if should_stop.item() == 1:
                break
            dist.barrier()

    if is_main:
        print(f"[DONE] Best val_loss={early_stop.best:.4f}")
        print(f"[DONE] Checkpoint: {CKPT_DIR / 'best.pth'}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DEF-UAVDETR Mega 8-GPU training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU")
    parser.add_argument(
        "--lr", type=float, default=5e-6,
        help="Base LR (auto-scaled by world_size)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--source", type=str, default="mega",
        choices=["mega", "unified"],
        help="mega=Nighthawk Mega, unified=unified_v2 fallback",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    import traceback as tb
    try:
        main()
    except Exception:
        tb.print_exc()
        raise
