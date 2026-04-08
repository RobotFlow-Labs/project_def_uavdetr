"""End-to-end DEF-UAVDETR detector wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from .losses import HybridInnerCiouNwdLoss
from .models.backbone import ConvNormLayer, WTConvBackbone
from .models.decoder import RTDETRHead
from .models.encoder import SWSAIFIEncoder
from .models.neck import ECFRFN


class DefUavDetr(nn.Module):
    """Paper-inspired end-to-end detector assembly."""

    def __init__(self, *, num_classes: int = 1, num_queries: int = 300) -> None:
        super().__init__()
        self.backbone = WTConvBackbone()
        self.s5_projection = ConvNormLayer(512, 256, 1, 1, act="relu")
        self.encoder = SWSAIFIEncoder(dim=256, num_heads=8, hidden_channels=1024)
        self.neck = ECFRFN()
        self.decoder = RTDETRHead(
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=num_queries,
            num_heads=8,
            num_decoder_layers=3,
        )
        self.loss_fn = HybridInnerCiouNwdLoss(ciou_alpha=0.6, nwd_constant=12.8)

    def forward_features(
        self,
        images: torch.Tensor,
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
