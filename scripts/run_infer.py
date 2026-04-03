"""CLI inference entrypoint for DEF-UAVDETR."""

from __future__ import annotations

import argparse
from pathlib import Path

from anima_def_uavdetr.infer import DefUavDetrPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DEF-UAVDETR inference on an image or folder.")
    parser.add_argument("source", type=Path, help="Image file or directory")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DefUavDetrPredictor(checkpoint_path=args.checkpoint, device=args.device)
    outputs = predictor.predict_from_path(args.source, conf=args.conf)
    for path, detections in outputs.items():
        print(f"{path}: {len(detections)} detections")


if __name__ == "__main__":
    main()
