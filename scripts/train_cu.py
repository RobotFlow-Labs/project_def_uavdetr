#!/usr/bin/env python3
"""CUDA-Accelerated training for project_def_uavdetr.

Uses shared CUDA kernels: detection_ops + fused_image_preprocess
Loader auto-handles Python version + torch ABI mismatches via JIT recompilation.

Usage:
    cd /mnt/forge-data/modules/03_wave7/project_def_uavdetr
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=3 python scripts/train_cu.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Setup paths
_root = Path(__file__).resolve().parent.parent
os.chdir(str(_root))
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "scripts"))
sys.path.insert(0, "/mnt/forge-data/shared_infra/cuda_extensions")

# Load CUDA kernels via universal loader (handles Python/torch version mismatches)
from load_kernel import load_cuda_kernel

_cu_loaded = {}
for _kname in ["detection_ops", "fused_image_preprocess"]:
    _mod = load_cuda_kernel(_kname)
    if _mod is not None:
        _cu_loaded[_kname] = _mod
        sys.modules[_kname] = _mod

print(f"[project_def_uavdetr-CU] Loaded CUDA kernels: {list(_cu_loaded.keys())}")

# Run the original training script
# No scripts/train.py main() found — try common patterns
_ran = False
try:
    import train as _train
    if hasattr(_train, "main"):
        _train.main()
        _ran = True
except (ImportError, ModuleNotFoundError):
    pass

if not _ran:
    print("[CU] CUDA kernels loaded and available on sys.path.")
    print("[CU] Run your training command manually with these kernels pre-loaded.")
    print(f"[CU] Available: {list(_cu_loaded.keys())}")
