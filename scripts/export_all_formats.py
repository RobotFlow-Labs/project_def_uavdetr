"""Complete export pipeline for DEF-UAVDETR.

Exports best.pth to all deployment formats: pth, safetensors, ONNX,
TorchScript, and TensorRT (if available). Generates HuggingFace model
card, copies paper/configs/logs, and pushes to HuggingFace.

Usage:
    uv run python scripts/export_all_formats.py \
        --checkpoint /mnt/artifacts-datai/checkpoints/project_def_uavdetr/best.pth \
        --device cuda:0

    # nohup-safe:
    nohup .venv/bin/python scripts/export_all_formats.py \
        --checkpoint /mnt/artifacts-datai/checkpoints/project_def_uavdetr/best.pth \
        > /mnt/artifacts-datai/logs/project_def_uavdetr/export.log 2>&1 & disown
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anima_def_uavdetr.checkpoints import load_checkpoint  # noqa: E402
from anima_def_uavdetr.model import DefUavDetr  # noqa: E402

logger = logging.getLogger("export_all_formats")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "ilessio-aiflowlab/project_def_uavdetr"
EXPORT_DIR = Path("/mnt/artifacts-datai/exports/project_def_uavdetr")
MODEL_DIR = Path("/mnt/artifacts-datai/models/project_def_uavdetr")


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_pytorch(model: DefUavDetr, output_dir: Path) -> Path:
    """Save raw PyTorch checkpoint."""
    path = output_dir / "def_uavdetr.pth"
    torch.save({"model": model.state_dict()}, path)
    logger.info("Exported PyTorch: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def export_safetensors(model: DefUavDetr, output_dir: Path) -> Path | None:
    """Export to safetensors format."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        logger.warning("safetensors not installed, skipping safetensors export")
        return None

    path = output_dir / "def_uavdetr.safetensors"
    save_file(model.state_dict(), str(path))
    logger.info("Exported safetensors: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def export_onnx(
    model: DefUavDetr,
    output_dir: Path,
    device: torch.device,
    image_size: int = 640,
) -> Path | None:
    """Export to ONNX with opset 18 (col2im support)."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        logger.warning("onnx not installed, skipping ONNX export")
        return None

    path = output_dir / "def_uavdetr.onnx"
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    try:
        torch.onnx.export(
            model_cpu,
            dummy,
            str(path),
            input_names=["images"],
            output_names=["pred_boxes", "pred_logits"],
            opset_version=18,
            dynamic_axes={
                "images": {0: "batch_size"},
                "pred_boxes": {0: "batch_size"},
                "pred_logits": {0: "batch_size"},
            },
            do_constant_folding=True,
        )
        logger.info("Exported ONNX (opset 18): %s (%.1f MB)", path, path.stat().st_size / 1e6)

        # Validate ONNX model
        import onnx

        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
    except Exception as e:
        logger.error("ONNX export failed: %s", e)
        if path.exists():
            path.unlink()
        return None
    finally:
        model.to(device)

    return path


def export_torchscript(
    model: DefUavDetr,
    output_dir: Path,
    device: torch.device,
    image_size: int = 640,
) -> Path | None:
    """Export to TorchScript via tracing."""
    path = output_dir / "def_uavdetr.torchscript.pt"
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    try:
        with torch.no_grad():
            traced = torch.jit.trace(model_cpu, dummy)
        traced.save(str(path))
        logger.info("Exported TorchScript: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    except Exception as e:
        logger.error("TorchScript export failed: %s", e)
        if path.exists():
            path.unlink()
        return None
    finally:
        model.to(device)

    return path


def export_tensorrt(
    onnx_path: Path,
    output_dir: Path,
    precision: str = "fp16",
    image_size: int = 640,
) -> Path | None:
    """Build TensorRT engine from ONNX. Requires tensorrt installed."""
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        logger.warning("tensorrt not installed, skipping TRT %s export", precision)
        return None

    suffix = f".{precision}.engine"
    path = output_dir / f"def_uavdetr{suffix}"

    try:
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("TRT parse error: %s", parser.get_error(i))
                return None

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("TRT: FP16 mode enabled")

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "images",
            min=(1, 3, image_size, image_size),
            opt=(1, 3, image_size, image_size),
            max=(4, 3, image_size, image_size),
        )
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            logger.error("TRT engine build failed for %s", precision)
            return None

        path.write_bytes(engine_bytes)
        size_mb = path.stat().st_size / 1e6
        logger.info("Exported TensorRT %s: %s (%.1f MB)", precision, path, size_mb)
        return path
    except Exception as e:
        logger.error("TensorRT %s export failed: %s", precision, e)
        return None


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def generate_model_card(output_dir: Path, exported_formats: list[str]) -> Path:
    """Generate HuggingFace model card README.md."""
    card = f"""---
license: apache-2.0
tags:
  - object-detection
  - anti-drone
  - uav-detection
  - rt-detr
  - defense
  - anima
datasets:
  - custom
pipeline_tag: object-detection
---

# DEF-UAVDETR -- Anti-Drone Detection

**Paper**: UAV-DETR: Anti-Drone Detection (arXiv: 2603.22841)

**Part of**: ANIMA Intelligence Compiler Suite by Robot Flow Labs

## Model Description

DEF-UAVDETR is a real-time end-to-end transformer-based detector for anti-drone
(UAV) detection. The architecture combines:

- **WTConv Backbone**: Wavelet Transform convolution for multi-scale feature extraction
- **SWSA-IFI Encoder**: Shifted Window Self-Attention with Inter-scale Feature Interaction
- **ECFRFN Neck**: Enhanced Cross-scale Feature Refinement and Fusion Network
- **RT-DETR Head**: Real-Time Detection Transformer decoder with 300 queries

## Specifications

| Property | Value |
|----------|-------|
| Parameters | 11.5M |
| Input Size | 640x640 |
| Classes | 1 (UAV) |
| Backbone | WTConv |
| Queries | 300 |

## Paper Targets

| Metric | Value |
|--------|-------|
| mAP50:95 | 62.56% |
| Precision | 96.82% |
| Recall | 94.93% |

## Available Formats

{chr(10).join(f"- {fmt}" for fmt in exported_formats)}

## Usage

```python
from anima_def_uavdetr.infer import DefUavDetrPredictor

predictor = DefUavDetrPredictor(
    checkpoint_path="def_uavdetr.pth",
    device="cuda:0",
)
detections = predictor.predict("image.jpg", conf=0.25)
# detections: list of [x1, y1, x2, y2, score, class_id] tensors
```

## Training

Trained on unified UAV dataset (236K images) with AdamW optimizer,
cosine LR schedule, 100 epochs, FP16 mixed precision.

## License

Apache 2.0

## Citation

```bibtex
@article{{uavdetr2025,
  title={{UAV-DETR: Anti-Drone Detection}},
  year={{2025}},
  journal={{arXiv preprint arXiv:2603.22841}}
}}
```

Built with ANIMA by Robot Flow Labs
"""
    path = output_dir / "README.md"
    path.write_text(card)
    logger.info("Generated model card: %s", path)
    return path


# ---------------------------------------------------------------------------
# Copy supporting files
# ---------------------------------------------------------------------------


def copy_supporting_files(output_dir: Path) -> None:
    """Copy paper, configs, and other supporting files to the export dir."""
    # Configs
    configs_src = PROJECT_ROOT / "configs"
    configs_dst = output_dir / "configs"
    if configs_src.is_dir():
        if configs_dst.exists():
            shutil.rmtree(configs_dst)
        shutil.copytree(configs_src, configs_dst)
        logger.info("Copied configs/")

    # Paper PDFs
    papers_src = PROJECT_ROOT / "papers"
    papers_dst = output_dir / "papers"
    if papers_src.is_dir():
        if papers_dst.exists():
            shutil.rmtree(papers_dst)
        shutil.copytree(papers_src, papers_dst)
        logger.info("Copied papers/")

    # Training logs (latest)
    logs_src = Path("/mnt/artifacts-datai/logs/project_def_uavdetr")
    if logs_src.is_dir():
        logs_dst = output_dir / "training_logs"
        logs_dst.mkdir(parents=True, exist_ok=True)
        log_files = sorted(logs_src.glob("*.log"))
        for lf in log_files[-3:]:  # Copy last 3 log files
            shutil.copy2(lf, logs_dst / lf.name)
        logger.info("Copied training logs")


# ---------------------------------------------------------------------------
# HuggingFace push
# ---------------------------------------------------------------------------


def push_to_huggingface(output_dir: Path, repo_id: str) -> bool:
    """Push exported model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            commit_message="[DEF-UAVDETR] Export trained model + all formats",
        )
        logger.info("Pushed to HuggingFace: https://huggingface.co/%s", repo_id)
        return True
    except ImportError:
        logger.warning("huggingface_hub not installed, skipping HF push")
        return False
    except Exception as e:
        logger.error("HuggingFace push failed: %s", e)
        # Try CLI fallback
        try:
            cmd = ["huggingface-cli", "upload", repo_id, str(output_dir), ".", "--private"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Pushed to HuggingFace via CLI: %s", repo_id)
            return True
        except Exception as e2:
            logger.error("CLI fallback also failed: %s", e2)
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complete export pipeline for DEF-UAVDETR",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/artifacts-datai/checkpoints/project_def_uavdetr/best.pth"),
        help="Path to best checkpoint",
    )
    p.add_argument("--device", type=str, default="cuda:0", help="Device")
    p.add_argument("--image-size", type=int, default=640, help="Input image size")
    p.add_argument(
        "--output",
        type=Path,
        default=MODEL_DIR,
        help="Output directory for all exports",
    )
    p.add_argument(
        "--hf-repo",
        type=str,
        default=HF_REPO,
        help="HuggingFace repo ID for upload",
    )
    p.add_argument("--no-push", action="store_true", help="Skip HuggingFace push")
    p.add_argument("--no-trt", action="store_true", help="Skip TensorRT export")
    p.add_argument("--no-torchscript", action="store_true", help="Skip TorchScript export")
    p.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info("=" * 60)
    logger.info("DEF-UAVDETR Export Pipeline")
    logger.info("=" * 60)

    # Validate checkpoint
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    # Device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = torch.device("cpu")

    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Device: %s", device)
    logger.info("Output: %s", args.output)

    # Load model
    model = DefUavDetr(num_classes=1, num_queries=300)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %.2fM parameters", param_count / 1e6)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Track exported formats
    exported_formats: list[str] = []
    export_manifest: dict[str, str | None] = {}
    t0 = time.time()

    # 1. PyTorch .pth
    logger.info("--- [1/6] PyTorch .pth ---")
    pth_path = export_pytorch(model, args.output)
    exported_formats.append("PyTorch (.pth)")
    export_manifest["pth"] = str(pth_path)

    # 2. safetensors
    logger.info("--- [2/6] safetensors ---")
    st_path = export_safetensors(model, args.output)
    if st_path:
        exported_formats.append("safetensors (.safetensors)")
        export_manifest["safetensors"] = str(st_path)
    else:
        export_manifest["safetensors"] = None

    # 3. ONNX
    onnx_path = None
    if not args.no_onnx:
        logger.info("--- [3/6] ONNX (opset 18) ---")
        onnx_path = export_onnx(model, args.output, device, args.image_size)
        if onnx_path:
            exported_formats.append("ONNX (.onnx, opset 18)")
            export_manifest["onnx"] = str(onnx_path)
        else:
            export_manifest["onnx"] = None
    else:
        logger.info("--- [3/6] ONNX: SKIPPED ---")
        export_manifest["onnx"] = None

    # 4. TorchScript
    if not args.no_torchscript:
        logger.info("--- [4/6] TorchScript ---")
        ts_path = export_torchscript(model, args.output, device, args.image_size)
        if ts_path:
            exported_formats.append("TorchScript (.torchscript.pt)")
            export_manifest["torchscript"] = str(ts_path)
        else:
            export_manifest["torchscript"] = None
    else:
        logger.info("--- [4/6] TorchScript: SKIPPED ---")
        export_manifest["torchscript"] = None

    # 5. TensorRT FP16
    if not args.no_trt and onnx_path is not None:
        logger.info("--- [5/6] TensorRT FP16 ---")
        trt16_path = export_tensorrt(onnx_path, args.output, "fp16", args.image_size)
        if trt16_path:
            exported_formats.append("TensorRT FP16 (.fp16.engine)")
            export_manifest["tensorrt_fp16"] = str(trt16_path)
        else:
            export_manifest["tensorrt_fp16"] = None
    else:
        logger.info("--- [5/6] TensorRT FP16: SKIPPED ---")
        export_manifest["tensorrt_fp16"] = None

    # 6. TensorRT FP32
    if not args.no_trt and onnx_path is not None:
        logger.info("--- [6/6] TensorRT FP32 ---")
        trt32_path = export_tensorrt(onnx_path, args.output, "fp32", args.image_size)
        if trt32_path:
            exported_formats.append("TensorRT FP32 (.fp32.engine)")
            export_manifest["tensorrt_fp32"] = str(trt32_path)
        else:
            export_manifest["tensorrt_fp32"] = None
    else:
        logger.info("--- [6/6] TensorRT FP32: SKIPPED ---")
        export_manifest["tensorrt_fp32"] = None

    elapsed = time.time() - t0

    # Generate model card
    logger.info("--- Generating model card ---")
    generate_model_card(args.output, exported_formats)

    # Copy supporting files
    logger.info("--- Copying supporting files ---")
    copy_supporting_files(args.output)

    # Save export manifest
    manifest = {
        "model": "DEF-UAVDETR",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint_source": str(args.checkpoint),
        "parameters": param_count,
        "image_size": args.image_size,
        "num_classes": 1,
        "num_queries": 300,
        "export_time_seconds": round(elapsed, 2),
        "formats": export_manifest,
        "exported_formats_list": exported_formats,
    }
    manifest_path = args.output / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Export manifest: %s", manifest_path)

    # Push to HuggingFace
    if not args.no_push:
        logger.info("--- Pushing to HuggingFace ---")
        push_to_huggingface(args.output, args.hf_repo)
    else:
        logger.info("--- HuggingFace push: SKIPPED ---")

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT PIPELINE COMPLETE -- DEF-UAVDETR")
    print("=" * 60)
    print(f"  Parameters:  {param_count / 1e6:.2f}M")
    print(f"  Export time:  {elapsed:.1f}s")
    print(f"  Output dir:   {args.output}")
    print()
    print(f"  {'Format':<30} {'Status':<10} {'Path'}")
    print(f"  {'-' * 70}")
    for fmt_key, fmt_path in export_manifest.items():
        status = "OK" if fmt_path else "SKIPPED"
        display_path = Path(fmt_path).name if fmt_path else "-"
        print(f"  {fmt_key:<30} {status:<10} {display_path}")
    print("=" * 60)

    if not args.no_push:
        print(f"  HF Repo: https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
