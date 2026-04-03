"""Export entrypoints for DEF-UAVDETR."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anima_def_uavdetr.checkpoints import load_checkpoint
from anima_def_uavdetr.model import DefUavDetr

SUPPORTED_EXPORTS = {
    "pytorch": ".pt",
    "onnx": ".onnx",
    "tensorrt": ".engine",
    "mlx": ".npz",
}


def export_pytorch(model: DefUavDetr, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)
    return path


def export_onnx(model: DefUavDetr, path: str | Path, *, image_size: int = 640) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        example_input,
        path,
        input_names=["images"],
        output_names=["pred_boxes", "pred_logits"],
        opset_version=17,
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DEF-UAVDETR weights.")
    parser.add_argument("--format", choices=sorted(SUPPORTED_EXPORTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = DefUavDetr().eval()
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")

    if args.format == "pytorch":
        export_pytorch(model, args.output)
    elif args.format == "onnx":
        export_onnx(model, args.output)
    else:
        raise NotImplementedError(f"{args.format} export is not implemented yet")


if __name__ == "__main__":
    main()
