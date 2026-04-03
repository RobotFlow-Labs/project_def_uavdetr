"""Release bundle generator for DEF-UAVDETR.

Creates a release manifest and directory structure suitable for
HuggingFace upload.

Usage:
    uv run python scripts/release.py --checkpoint best.pth --output release/
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from anima_def_uavdetr.version import __version__


def build_release_manifest(
    *,
    checkpoint_path: str | None = None,
    metrics: dict | None = None,
    benchmark: dict | None = None,
) -> dict:
    """Build a release manifest dict."""
    return {
        "model": "def-uavdetr",
        "version": __version__,
        "paper": "UAV-DETR: Anti-Drone Detection",
        "arxiv": "2603.22841",
        "timestamp": datetime.now(UTC).isoformat(),
        "paper_targets": {
            "custom_uav_map50_95": 62.56,
            "dut_anti_uav_map50_95": 67.15,
            "model_params": 11_962_040,
        },
        "achieved_metrics": metrics or {},
        "benchmark": benchmark or {},
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "artifacts": {
            "pytorch": "model.pth",
            "safetensors": "model.safetensors",
            "onnx": "model.onnx",
            "tensorrt_fp16": "model.fp16.engine",
            "tensorrt_fp32": "model.fp32.engine",
        },
        "limits": {
            "known_false_positives": ["bird-like distractors", "kites", "balloons"],
            "known_false_negatives": ["urban camouflage", "heavy occlusion"],
            "flops_overhead_pct": 17.2,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DEF-UAVDETR release builder")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--metrics", type=Path, default=None, help="JSON with eval metrics")
    parser.add_argument("--benchmark", type=Path, default=None, help="JSON with benchmark results")
    parser.add_argument("--output", type=Path, default=Path("release"))
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text()) if args.metrics else None
    bench = json.loads(args.benchmark.read_text()) if args.benchmark else None

    manifest = build_release_manifest(
        checkpoint_path=args.checkpoint,
        metrics=metrics,
        benchmark=bench,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Release manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
