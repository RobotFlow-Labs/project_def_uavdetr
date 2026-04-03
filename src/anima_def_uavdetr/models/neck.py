"""ECFRFN neck for DEF-UAVDETR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvNormLayer


def upsample_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ResidualBottleneck(nn.Module):
    """Lightweight stand-in for the RepNCSP bottleneck path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cv1 = ConvNormLayer(channels, channels, 3, 1, act="relu")
        self.cv2 = ConvNormLayer(channels, channels, 3, 1, act=None)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.cv2(self.cv1(x)))


class RepNCSP(nn.Module):
    """Minimal CSP block used inside RepNCSPELAN4."""

    def __init__(self, c1: int, c2: int, depth: int = 1) -> None:
        super().__init__()
        self.cv1 = ConvNormLayer(c1, c2, 1, 1, act="relu")
        self.blocks = nn.Sequential(*[ResidualBottleneck(c2) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.cv1(x))


class RepNCSPELAN4(nn.Module):
    """Channel mixing block adapted from the reference neck."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, c5: int = 1) -> None:
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer(c1, c3, 1, 1, act="relu")
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), ConvNormLayer(c4, c4, 3, 1, act="relu"))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), ConvNormLayer(c4, c4, 3, 1, act="relu"))
        self.cv4 = ConvNormLayer(c3 + (2 * c4), c2, 1, 1, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(module(y[-1]) for module in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class SBA(nn.Module):
    """Selective bilateral aggregation from the reference implementation."""

    def __init__(self, in_channels: tuple[int, int], input_dim: int = 64) -> None:
        super().__init__()
        self.d_in1 = ConvNormLayer(input_dim // 2, input_dim // 2, 1, 1, act="relu")
        self.d_in2 = ConvNormLayer(input_dim // 2, input_dim // 2, 1, 1, act="relu")
        self.conv = ConvNormLayer(input_dim, input_dim, 3, 1, act="relu")
        self.fc1 = nn.Conv2d(in_channels[1], input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels[0], input_dim // 2, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        high_feature, low_feature = x

        low_feature = self.fc1(low_feature)
        high_feature = self.fc2(high_feature)

        gate_low = self.sigmoid(low_feature)
        gate_high = self.sigmoid(high_feature)

        low_feature = self.d_in1(low_feature)
        high_feature = self.d_in2(high_feature)

        low_feature = low_feature + low_feature * gate_low + (1 - gate_low) * upsample_to(
            gate_high * high_feature, low_feature.shape[-2:]
        )
        high_feature = high_feature + high_feature * gate_high + (1 - gate_high) * upsample_to(
            gate_low * low_feature, high_feature.shape[-2:]
        )

        high_feature = upsample_to(high_feature, low_feature.shape[-2:])
        return self.conv(torch.cat([high_feature, low_feature], dim=1))


class ECFRFN(nn.Module):
    """Cross-scale fusion neck that emits P2..P5 with 256 channels."""

    def __init__(self) -> None:
        super().__init__()
        self.lateral_p5 = ConvNormLayer(256, 256, 1, 1, act="relu")

        self.fuse_s4 = SBA((256, 256))
        self.fuse_s3 = SBA((256, 128))
        self.fuse_s2 = SBA((256, 64))
        self.refine_p3 = SBA((256, 256))
        self.refine_p4 = SBA((256, 256))
        self.refine_p5 = SBA((256, 256))

        self.block_p4 = RepNCSPELAN4(64, 256, 128, 64, 1)
        self.block_p3 = RepNCSPELAN4(64, 256, 128, 64, 1)
        self.block_p2 = RepNCSPELAN4(64, 256, 128, 64, 1)
        self.block_p3_refined = RepNCSPELAN4(64, 256, 128, 64, 1)
        self.block_p4_refined = RepNCSPELAN4(64, 256, 128, 64, 1)
        self.block_p5_refined = RepNCSPELAN4(64, 256, 128, 64, 1)

    def forward(
        self,
        s2: torch.Tensor,
        s3: torch.Tensor,
        s4: torch.Tensor,
        f5: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p5_seed = self.lateral_p5(f5)
        p4 = self.block_p4(self.fuse_s4((p5_seed, s4)))
        p3 = self.block_p3(self.fuse_s3((p4, s3)))
        p2 = self.block_p2(self.fuse_s2((p3, s2)))

        p3_refined = self.block_p3_refined(self.refine_p3((p2, p3)))
        p4_refined = self.block_p4_refined(self.refine_p4((p3_refined, p4)))
        p5_refined = self.block_p5_refined(self.refine_p5((p4_refined, p5_seed)))
        return p2, p3_refined, p4_refined, p5_refined
