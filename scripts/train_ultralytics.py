"""Train UAV-DETR using the Ultralytics RT-DETR pipeline.

Uses the battle-tested Ultralytics training with all RT-DETR components:
- Deformable cross-attention
- Hungarian matcher + VarifocalLoss
- Encoder-guided query initialization
- Auxiliary decoder layer losses
- Denoising training

Usage (1 GPU):
    CUDA_VISIBLE_DEVICES=7 uv run python scripts/train_ultralytics.py --epochs 10 --batch 4

Usage (4 GPUs):
    CUDA_VISIBLE_DEVICES=0,1,4,7 uv run python scripts/train_ultralytics.py --epochs 100 --batch 16
"""

from __future__ import annotations

import argparse
import os
import sys

# Add the reference repo to path
REPO_DIR = os.path.join(os.path.dirname(__file__), "..", "repositories", "UAVDETR")
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)


def main():
    parser = argparse.ArgumentParser(description="Train UAV-DETR via Ultralytics")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--data", type=str, default="datasets/unified/data.yaml")
    parser.add_argument("--model", type=str, default="ultralytics/cfg/models/UAV-DETR.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="/mnt/artifacts-datai/checkpoints/project_def_uavdetr/ultralytics")
    parser.add_argument("--name", type=str, default="uavdetr_v3")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    from ultralytics import RTDETR

    model = RTDETR(args.model)

    if args.resume:
        model.load(args.resume)

    model.train(
        data=args.data,
        cache=False,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=0,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
