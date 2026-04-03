"""Runtime benchmark harness for DEF-UAVDETR.

Paper reference: Section 4.6 — measures latency and throughput on the
target hardware.

Usage:
    uv run python scripts/benchmark.py --device cpu --iterations 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from anima_def_uavdetr.infer import DefUavDetrPredictor


def benchmark(
    *,
    device: str = "cpu",
    image_size: int = 640,
    iterations: int = 100,
    warmup: int = 5,
    checkpoint: str | None = None,
) -> dict:
    """Run inference benchmark and return timing stats."""
    predictor = DefUavDetrPredictor(
        checkpoint_path=checkpoint,
        device=device,
    )

    dummy = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        predictor.predict(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        predictor.predict(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies)
    params = sum(p.numel() for p in predictor.model.parameters())

    return {
        "device": device,
        "image_size": image_size,
        "iterations": iterations,
        "warmup": warmup,
        "parameters": params,
        "mean_ms": round(float(latencies_arr.mean()), 2),
        "std_ms": round(float(latencies_arr.std()), 2),
        "min_ms": round(float(latencies_arr.min()), 2),
        "max_ms": round(float(latencies_arr.max()), 2),
        "p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
        "throughput_fps": round(1000.0 / float(latencies_arr.mean()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DEF-UAVDETR benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = benchmark(
        device=args.device,
        image_size=args.image_size,
        iterations=args.iterations,
        warmup=args.warmup,
        checkpoint=args.checkpoint,
    )

    print(json.dumps(results, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
