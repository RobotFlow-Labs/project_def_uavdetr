"""WTConv backbone used by DEF-UAVDETR."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act: str | nn.Module | None) -> nn.Module:
    """Return an activation module from a short name."""

    if isinstance(act, nn.Module):
        return act
    if act is None:
        return nn.Identity()
    normalized = act.lower()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    if normalized == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    raise ValueError(f"unsupported activation: {act}")


class ConvNormLayer(nn.Module):
    """Convolution + batch norm + activation."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int,
        padding: int | None = None,
        *,
        bias: bool = False,
        act: str | nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _ScaleModule(nn.Module):
    def __init__(self, dims: Sequence[int], init_scale: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x


def create_wavelet_filter(
    wave: str,
    in_size: int,
    out_size: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create analysis and synthesis filters for the chosen wavelet."""

    wavelet = pywt.Wavelet(wave)
    dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=dtype)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(wavelet.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_lo = torch.tensor(wavelet.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters.to(x.device), stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters.to(x.device), stride=2, groups=c, padding=pad)


class WTConv2d(nn.Module):
    """Wavelet convolution adapted from the reference implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        stride: int = 1,
        bias: bool = True,
        wt_levels: int = 1,
        wt_type: str = "db1",
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("WTConv2d requires in_channels == out_channels")

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride

        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels)
        self.register_buffer("wt_filter", wt_filter, persistent=False)
        self.register_buffer("iwt_filter", iwt_filter, persistent=False)

        self.base_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            stride=1,
            groups=in_channels,
            bias=bias,
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size,
                    padding="same",
                    stride=1,
                    groups=in_channels * 4,
                    bias=False,
                )
                for _ in range(wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(wt_levels)]
        )

        if stride > 1:
            stride_filter = torch.ones(in_channels, 1, 1, 1)
            self.register_buffer("stride_filter", stride_filter, persistent=False)
        else:
            self.register_buffer("stride_filter", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll_in_levels: list[torch.Tensor] = []
        x_h_in_levels: list[torch.Tensor] = []
        shapes_in_levels: list[torch.Size] = []
        curr_x_ll = x

        for level in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if curr_shape[2] % 2 or curr_shape[3] % 2:
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[level](self.wavelet_convs[level](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll: torch.Tensor | int = 0
        for _ in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        x_tag = next_x_ll
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        if self.stride > 1:
            x = F.conv2d(x, self.stride_filter.to(x.device), stride=self.stride, groups=self.in_channels)
        return x


class BasicBlock(nn.Module):
    """Residual block used in the RT-DETR PResNet stem."""

    expansion = 1

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        *,
        stride: int,
        shortcut: bool,
        act: str = "relu",
        variant: str = "d",
    ) -> None:
        super().__init__()
        self.shortcut = shortcut

        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch2a(x)
        out = self.branch2b(out)
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class BasicBlockWTConv(BasicBlock):
    """BasicBlock whose second branch uses WTConv."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        *,
        stride: int,
        shortcut: bool,
        act: str = "relu",
        variant: str = "d",
    ) -> None:
        super().__init__(ch_in, ch_out, stride=stride, shortcut=shortcut, act=act, variant=variant)
        self.branch2b = WTConv2d(ch_out, ch_out)


class StageBlocks(nn.Module):
    """Stage wrapper matching the reference stride pattern."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        *,
        count: int,
        stage_num: int,
        act: str = "relu",
    ) -> None:
        super().__init__()
        blocks = []
        for index in range(count):
            blocks.append(
                BasicBlockWTConv(
                    ch_in,
                    ch_out,
                    stride=2 if index == 0 and stage_num != 2 else 1,
                    shortcut=False if index == 0 else True,
                    act=act,
                )
            )
            if index == 0:
                ch_in = ch_out
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class WTConvBackbone(nn.Module):
    """Paper-faithful backbone emitting S2..S5."""

    def __init__(self) -> None:
        super().__init__()
        self.stem0 = ConvNormLayer(3, 32, 3, 2, act="relu")
        self.stem1 = ConvNormLayer(32, 32, 3, 1, act="relu")
        self.stem2 = ConvNormLayer(32, 64, 3, 1, act="relu")
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = StageBlocks(64, 64, count=2, stage_num=2, act="relu")
        self.stage3 = StageBlocks(64, 128, count=2, stage_num=3, act="relu")
        self.stage4 = StageBlocks(128, 256, count=2, stage_num=4, act="relu")
        self.stage5 = StageBlocks(256, 512, count=2, stage_num=5, act="relu")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem0(x)
        x = self.stem1(x)
        x = self.stem2(x)
        s2 = self.pool(x)
        s2 = self.stage2(s2)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return s2, s3, s4, s5
