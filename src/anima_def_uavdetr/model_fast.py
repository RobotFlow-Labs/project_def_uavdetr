"""Optimized DEF-UAVDETR with CUDA kernels.

Drop-in replacement for ``model.DefUavDetr`` that uses:
- Fused wavelet CUDA kernel (backbone)
- Deformable attention CUDA kernel (encoder)

State dict is compatible with the original model — can hot-swap from
a checkpoint trained with the original model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .losses import HybridInnerCiouNwdLoss
from .models.backbone import ConvNormLayer, WTConvBackbone
from .models.backbone_fast import (
    _HAS_FUSED_WAVELET,
    inverse_wavelet_transform_fast,
    wavelet_transform_fast,
)
from .models.decoder import RTDETRHead
from .models.encoder_fast import SWSAIFIEncoderFast, _HAS_DEFORM_ATTN
from .models.neck import ECFRFN


def _patch_backbone_wavelets(backbone: WTConvBackbone) -> None:
    """Monkey-patch backbone WTConv layers to use fused CUDA kernels."""
    if not _HAS_FUSED_WAVELET:
        return
    for module in backbone.modules():
        if hasattr(module, "wt_function"):
            module.wt_function = wavelet_transform_fast
        if hasattr(module, "iwt_function"):
            module.iwt_function = inverse_wavelet_transform_fast


class DefUavDetrFast(nn.Module):
    """Optimized UAV-DETR with CUDA kernels.

    Compatible state dict with ``DefUavDetr`` — can load checkpoints
    trained with the original model.
    """

    def __init__(self, *, num_classes: int = 1, num_queries: int = 300) -> None:
        super().__init__()
        self.backbone = WTConvBackbone()
        self.s5_projection = ConvNormLayer(512, 256, 1, 1, act="relu")
        # Use fast encoder (deformable attn if available)
        self.encoder = SWSAIFIEncoderFast(dim=256, num_heads=8, hidden_channels=1024)
        self.neck = ECFRFN()
        self.decoder = RTDETRHead(
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=num_queries,
            num_heads=8,
            num_decoder_layers=3,
        )
        self.loss_fn = HybridInnerCiouNwdLoss(alpha=0.6, nwd_constant=12.8)

        # Patch backbone with fused wavelet kernels
        _patch_backbone_wavelets(self.backbone)

        self._optimizations = []
        if _HAS_FUSED_WAVELET:
            self._optimizations.append("fused_wavelet")
        if _HAS_DEFORM_ATTN:
            self._optimizations.append("deformable_attn")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load with automatic Conv2d→Linear weight reshaping for encoder FFN."""
        adapted = {}
        for k, v in state_dict.items():
            if "encoder.block.fc" in k and "weight" in k and v.dim() == 4:
                # Conv2d [out, in, 1, 1] → Linear [out, in]
                adapted[k] = v.squeeze(-1).squeeze(-1)
            elif "encoder.block.norm" in k and "weight" in k and v.dim() == 1:
                adapted[k] = v
            else:
                adapted[k] = v
        return super().load_state_dict(adapted, strict=strict, assign=assign)

    def forward_features(
        self, images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s2, s3, s4, s5 = self.backbone(images)
        f5 = self.encoder(self.s5_projection(s5))
        return self.neck(s2, s3, s4, f5)

    def forward(
        self,
        images: torch.Tensor,
        targets: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        pyramid = self.forward_features(images)
        pred_boxes, pred_logits = self.decoder(pyramid, targets=targets)
        if targets is None:
            return pred_boxes, pred_logits
        return self.loss_fn(pred_boxes, pred_logits, targets)
