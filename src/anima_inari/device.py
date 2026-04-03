"""Dual-compute device abstraction — MLX + CUDA mandatory."""
import os

BACKEND = os.environ.get("ANIMA_BACKEND", "auto")

if BACKEND == "auto":
    try:
        import mlx.core as mx  # noqa: F401
        BACKEND = "mlx"
    except ImportError:
        try:
            import torch
            BACKEND = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            BACKEND = "cpu"


def get_backend() -> str:
    return BACKEND


def get_device():
    if BACKEND == "mlx":
        import mlx.core as mx
        return mx.default_device()
    elif BACKEND == "cuda":
        import torch
        return torch.device("cuda")
    else:
        import torch
        return torch.device("cpu")
