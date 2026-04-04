"""Fast WTConv backbone using fused CUDA wavelet kernels.

Drop-in replacement for wavelet_transform / inverse_wavelet_transform.
Falls back to PyTorch implementation when CUDA kernel is unavailable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

# Try to load fused CUDA wavelet kernel
_HAS_FUSED_WAVELET = False
try:
    import fused_wavelet_cuda

    _HAS_FUSED_WAVELET = True
except ImportError:
    fused_wavelet_cuda = None


def wavelet_transform_fast(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """DWT using fused CUDA kernel when available."""
    if _HAS_FUSED_WAVELET and x.is_cuda:
        return fused_wavelet_cuda.fused_dwt(x)
    # Fallback: original PyTorch path
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters.to(x.device), stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_wavelet_transform_fast(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """IDWT using fused CUDA kernel when available."""
    if _HAS_FUSED_WAVELET and x.is_cuda:
        b, c, _, h_half, w_half = x.shape
        return fused_wavelet_cuda.fused_idwt(x, h_half * 2, w_half * 2)
    # Fallback: original PyTorch path
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters.to(x.device), stride=2, groups=c, padding=pad)
