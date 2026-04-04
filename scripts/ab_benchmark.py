"""A/B benchmark: original vs CUDA-optimized model.

Loads best.pth from current training, runs both models side-by-side,
reports speedup. If optimized model is faster, prints the command to
hot-swap training.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/ab_benchmark.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")

CKPT_PATH = "/mnt/artifacts-datai/checkpoints/project_def_uavdetr/best.pth"
BATCH_SIZE = 16
IMG_SIZE = 640
N_WARMUP = 5
N_ITERS = 20


def bench_model(model, device, label: str) -> float:
    """Run forward+backward benchmark, return avg ms/step."""
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)
    targets = {
        "boxes": [torch.tensor([[0.5, 0.5, 0.1, 0.1]]).to(device) for _ in range(BATCH_SIZE)],
        "labels": [torch.tensor([0]).to(device) for _ in range(BATCH_SIZE)],
    }

    # Warmup
    for _ in range(N_WARMUP):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ld = model(dummy, targets)
            loss = sum(ld.values())
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for i in range(N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ld = model(dummy, targets)
            loss = sum(ld.values())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)

    avg = np.mean(times)
    std = np.std(times)
    vram = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"[{label}] {avg:.0f}ms/step (±{std:.0f}ms), {1000/avg:.1f} steps/s, VRAM={vram:.2f}GB")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return avg


def main():
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Batch size: {BATCH_SIZE}, Image: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Warmup: {N_WARMUP}, Iterations: {N_ITERS}")
    print()

    # ── Model A: Original ──
    from anima_def_uavdetr.model import DefUavDetr

    model_a = DefUavDetr(num_classes=1, num_queries=300)

    # Load checkpoint if exists
    try:
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        model_a.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        step = ckpt.get("step", "?")
        val_loss = ckpt.get("val_loss", "?")
        print(f"Loaded checkpoint: epoch={epoch}, step={step}, val_loss={val_loss}")
    except FileNotFoundError:
        print("No checkpoint found — using random weights")
    print()

    time_a = bench_model(model_a, device, "A: Original")
    del model_a
    torch.cuda.empty_cache()

    # ── Model B: Optimized ──
    from anima_def_uavdetr.model_fast import DefUavDetrFast

    model_b = DefUavDetrFast(num_classes=1, num_queries=300)
    print(f"Optimizations: {model_b._optimizations}")

    # Load same checkpoint (compatible state dict for shared layers)
    try:
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        missing, unexpected = model_b.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  Missing keys (new layers): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    except FileNotFoundError:
        pass
    print()

    time_b = bench_model(model_b, device, "B: Optimized")

    # ── Results ──
    print()
    speedup = time_a / time_b
    print(f"{'='*50}")
    print(f"  A (original):  {time_a:.0f} ms/step  ({1000/time_a:.1f} steps/s)")
    print(f"  B (optimized): {time_b:.0f} ms/step  ({1000/time_b:.1f} steps/s)")
    print(f"  Speedup:       {speedup:.2f}x")
    print(f"{'='*50}")

    if speedup > 1.1:
        hours_saved = (446100 - int(ckpt.get("step", 0))) / (1000 / time_b) / 3600
        hours_original = (446100 - int(ckpt.get("step", 0))) / (1000 / time_a) / 3600
        print(f"\n  B is {speedup:.1f}x faster!")
        print(f"  Remaining time: {hours_original:.1f}h (A) → {hours_saved:.1f}h (B)")
        print(f"  Time saved: {hours_original - hours_saved:.1f} hours")
        print(f"\n  To hot-swap, kill current training and run:")
        print(f"  CUDA_VISIBLE_DEVICES=7 nohup uv run python scripts/train_cuda.py \\")
        print(f"    --epochs 100 --batch-size {BATCH_SIZE} --datasets seraphim \\")
        print(f"    --resume {CKPT_PATH} --no-compile --model fast \\")
        print(f"    > /mnt/artifacts-datai/logs/project_def_uavdetr/train_fast.log 2>&1 &")
    else:
        print(f"\n  Speedup < 1.1x — not worth hot-swapping.")


if __name__ == "__main__":
    main()
