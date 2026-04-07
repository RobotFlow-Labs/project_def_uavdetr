"""Multi-GPU DDP training for DEF-UAVDETR.

Uses PyTorch DistributedDataParallel for linear scaling across 4x L4 GPUs.
Loads checkpoint from single-GPU run and continues finetuning.

Launch (4 GPUs):
    PYTHONUNBUFFERED=1 PYTHONPATH="" \
    nohup torchrun --nproc_per_node=4 --master_port=29500 \
        scripts/train_ddp.py \
        --epochs 30 --batch-size 16 --lr 1e-5 \
        --datasets seraphim,dut_anti_uav,visdrone \
        --resume /mnt/artifacts-datai/checkpoints/project_def_uavdetr/best.pth \
        > /mnt/artifacts-datai/logs/project_def_uavdetr/train_ddp_$(date +%Y%m%d_%H%M).log 2>&1 &
    disown

Launch (specific GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ...
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from anima_def_uavdetr.model import DefUavDetr

# ── paths ──────────────────────────────────────────────────────────────
CKPT_DIR = Path("/mnt/artifacts-datai/checkpoints/project_def_uavdetr")
LOG_DIR = Path("/mnt/artifacts-datai/logs/project_def_uavdetr")

DATASETS = {
    "seraphim": Path("/mnt/forge-data/datasets/uav_detection/seraphim"),
    "dut_anti_uav": Path("/mnt/train-data/datasets/dut_anti_uav"),
    "birddrone": Path("/mnt/forge-data/shared_infra/datasets/lat_birddrone"),
    "dronevehicle_night": Path("/mnt/forge-data/shared_infra/datasets/dronevehicle_night"),
    "visdrone": Path("/mnt/forge-data/datasets/uav_detection/visdrone"),
}

IMG_SIZE = 640


# ── dataset (same as single-GPU) ──────────────────────────────────────
class YoloDetectionDataset(Dataset):
    def __init__(
        self, image_dir: Path, label_dir: Path,
        img_size: int = 640, augment: bool = False,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in exts and p.is_file()
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_labels(self, stem: str) -> torch.Tensor:
        lbl_path = self.label_dir / f"{stem}.txt"
        if not lbl_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)
        try:
            data = np.loadtxt(str(lbl_path), dtype=np.float32).reshape(-1, 5)
            return torch.from_numpy(data)
        except (ValueError, IndexError):
            return torch.zeros((0, 5), dtype=torch.float32)

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
        labels = self._load_labels(img_path.stem)
        if labels.numel() == 0:
            target = {
                "labels": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }
        else:
            target = {"labels": labels[:, 0].long(), "boxes": labels[:, 1:]}
        return tensor, target


def build_datasets(dataset_names: list[str], split: str = "train", augment: bool = False):
    datasets = []
    for name in dataset_names:
        root = DATASETS.get(name)
        if root is None or not root.exists():
            continue
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            continue
        ds = YoloDetectionDataset(img_dir, lbl_dir, IMG_SIZE, augment=augment)
        if len(ds) > 0:
            datasets.append(ds)
    if not datasets:
        raise FileNotFoundError(f"No valid datasets found for split={split}")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


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


# ── checkpoint manager (rank 0 only) ──────────────────────────────────
class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, mode: str = "min"):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int) -> Path:
        path = self.save_dir / f"ddp_checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
        best_val, best_path = self.history[0]
        shutil.copy2(best_path, self.save_dir / "best.pth")
        return path


# ── early stopping ─────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
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


# ── DDP training loop ─────────────────────────────────────────────────
def train(args):
    # ── DDP setup ──────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = local_rank == 0

    if is_main:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[DDP] world_size={world_size}, device={device}")
        print(f"[DDP] effective_batch_size={args.batch_size * world_size}")
        props = torch.cuda.get_device_properties(device)
        print(f"[GPU] {props.name}, {props.total_memory / 1e9:.1f} GB VRAM")

    # ── model ──────────────────────────────────────────────────────
    model = DefUavDetr(num_classes=1, num_queries=300).to(device)
    if is_main:
        params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] DefUavDetr: {params:,} parameters")

    # ── load checkpoint (single-GPU format) ────────────────────────
    if args.resume and Path(args.resume).exists():
        if is_main:
            print(f"[RESUME] Loading {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)  # noqa: S301
        model.load_state_dict(ckpt["model"])
        if is_main:
            epoch_info = ckpt.get("epoch", "?")
            val_info = ckpt.get("val_loss", "?")
            print(f"[RESUME] Loaded epoch={epoch_info}, val_loss={val_info}")

    # ── wrap in DDP ────────────────────────────────────────────────
    model = DDP(model, device_ids=[local_rank])

    # ── data ───────────────────────────────────────────────────────
    dataset_names = args.datasets.split(",")
    full_train_ds = build_datasets(dataset_names, split="train", augment=True)

    # Split train/val (5% val)
    n_val = max(1, int(len(full_train_ds) * 0.05))
    n_train = len(full_train_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    if is_main:
        print(f"[DATA] datasets={dataset_names}")
        print(f"[DATA] train={len(train_ds)}, val={len(val_ds)}")

    # Distributed samplers
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

    # ── optimizer + scheduler ──────────────────────────────────────
    # Scale LR linearly with world_size
    scaled_lr = args.lr * world_size
    if is_main:
        print(f"[OPT] base_lr={args.lr}, scaled_lr={scaled_lr} (x{world_size})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    # ── managers (rank 0 only) ─────────────────────────────────────
    ckpt_mgr = CheckpointManager(CKPT_DIR, keep_top_k=2, mode="min") if is_main else None
    early_stop = EarlyStopping(patience=args.patience, min_delta=1e-4)

    if is_main:
        print(f"[TRAIN] epochs={args.epochs}, warmup={warmup_steps}, total_steps={total_steps}")
        print(f"[TRAIN] steps/epoch={len(train_loader)}")

    global_step = 0
    metrics_path = LOG_DIR / "metrics_ddp.jsonl"

    with open(metrics_path, "a") if is_main else open("/dev/null", "a") as metrics_file:
        for epoch in range(args.epochs):
            model.train()
            train_sampler.set_epoch(epoch)  # shuffle differently each epoch
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

                if torch.isnan(loss):
                    if is_main:
                        print(f"[FATAL] NaN loss at epoch {epoch}, step {global_step}")
                    dist.destroy_process_group()
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

                if is_main and global_step % 50 == 0:
                    lr = scheduler.get_lr()
                    sps = epoch_steps / (time.time() - t_epoch)
                    print(
                        f"[Step {global_step}] loss={loss_val:.4f} lr={lr:.2e} "
                        f"speed={sps:.1f} steps/s ({sps * world_size:.1f} eff)"
                    )

            # ── validation (all ranks compute, rank 0 aggregates) ──
            model.eval()
            val_loss_local = torch.tensor(0.0, device=device)
            val_steps_local = torch.tensor(0, device=device)
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
                    val_loss_local += loss
                    val_steps_local += 1

            # All-reduce val loss across ranks
            dist.all_reduce(val_loss_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_steps_local, op=dist.ReduceOp.SUM)

            avg_train = epoch_loss / max(epoch_steps, 1)
            avg_val = (val_loss_local / max(val_steps_local, 1)).item()
            epoch_time = time.time() - t_epoch
            lr = scheduler.get_lr()

            if is_main:
                print(
                    f"[Epoch {epoch + 1}/{args.epochs}] "
                    f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
                    f"lr={lr:.2e} time={epoch_time:.1f}s"
                )

                record = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train_loss": round(avg_train, 6),
                    "val_loss": round(avg_val, 6),
                    "lr": lr,
                    "epoch_time_s": round(epoch_time, 1),
                    "world_size": world_size,
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()

                # Checkpoint (rank 0 only, using unwrapped model)
                state = {
                    "model": model.module.state_dict(),
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

            # Early stopping (broadcast from rank 0)
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
        print(f"[DONE] Best checkpoint: {CKPT_DIR / 'best.pth'}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DEF-UAVDETR DDP training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size")
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Base LR (auto-scaled by world_size)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--datasets", type=str, default="seraphim,dut_anti_uav,visdrone",
        help="Comma-separated dataset names",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
