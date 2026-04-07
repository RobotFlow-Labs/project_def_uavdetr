"""GPU batch size finder for DEF-UAVDETR.

Runs binary search to find optimal batch size at target VRAM utilisation.
Uses a real forward+backward pass for accurate memory measurement.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/find_batch_size.py --target 0.65
"""

from __future__ import annotations

import argparse

import torch

from anima_def_uavdetr.model import DefUavDetr


def find_optimal_batch(
    target_util: float = 0.65,
    device_id: int = 0,
    img_size: int = 640,
) -> int:
    """Binary search for max batch size at target VRAM utilisation."""
    device = torch.device(f"cuda:{device_id}")
    model = DefUavDetr(num_classes=1, num_queries=300).to(device)
    model.train()

    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_bytes = int(total_mem * target_util)
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}, {total_mem / 1e9:.1f} GB")
    print(f"Target: {target_util * 100:.0f}% = {target_bytes / 1e9:.1f} GB")

    bs = 2
    best_bs = 2
    while bs <= 128:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            dummy = torch.randn(bs, 3, img_size, img_size, device=device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred_boxes, pred_logits = model(dummy)
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

    print(f"\nOptimal batch_size: {best_bs}")
    return best_bs


def main():
    parser = argparse.ArgumentParser(description="DEF-UAVDETR batch size finder")
    parser.add_argument("--target", type=float, default=0.65)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    find_optimal_batch(target_util=args.target, img_size=args.image_size)


if __name__ == "__main__":
    main()
