"""SWSA-IFI encoder for DEF-UAVDETR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class PatchSA(nn.Module):
    """Sliding-window patch self-attention from the reference implementation."""

    def __init__(self, dim: int, *, heads: int = 8, patch_size: int = 4, stride: int = 4) -> None:
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.patch_size = patch_size
        self.stride = stride

        self.to_qkv = nn.Conv2d(dim * 3, dim * 3, 1, groups=dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

        num_positions = (2 * patch_size - 1) ** 2
        self.pos_encode = nn.Parameter(torch.zeros(num_positions, heads))
        nn.init.trunc_normal_(self.pos_encode, std=0.02)

        coord = torch.arange(patch_size)
        coords = torch.stack(torch.meshgrid([coord, coord], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += patch_size - 1
        relative_coords[:, :, 1] += patch_size - 1
        relative_coords[:, :, 0] *= 2 * patch_size - 1
        self.register_buffer("pos_index", relative_coords.sum(-1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if h != w:
            raise ValueError("PatchSA expects square feature maps")

        pad_num = self.patch_size - self.stride
        patch_num = ((h + pad_num - self.patch_size) // self.stride + 1) ** 2
        expanded = F.pad(x, (0, pad_num, 0, pad_num), mode="replicate")
        qkv = self.to_qkv(torch.cat([expanded, expanded, expanded], dim=1))

        qkv_patches = F.unfold(qkv, kernel_size=self.patch_size, stride=self.stride)
        qkv_patches = qkv_patches.view(
            b, 3, self.heads, -1, self.patch_size**2, patch_num
        ).permute(1, 0, 2, 5, 4, 3)
        q, k, v = qkv_patches[0], qkv_patches[1], qkv_patches[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        pos_encode = self.pos_encode[self.pos_index.reshape(-1)].view(
            self.patch_size**2, self.patch_size**2, -1
        )
        pos_encode = pos_encode.permute(2, 0, 1).contiguous().unsqueeze(1).repeat(1, patch_num, 1, 1)
        attn = self.softmax(attn + pos_encode.unsqueeze(0))

        attended = attn @ v
        attended = attended.view(
            b, self.heads, patch_num, self.patch_size, self.patch_size, -1
        )[:, :, :, : self.stride, : self.stride]
        attended = attended.transpose(2, 5).contiguous().view(b, -1, patch_num)
        folded = F.fold(attended, output_size=(h, w), kernel_size=self.stride, stride=self.stride)
        return self.to_out(folded)


class TransformerEncoderLayerSWSA(nn.Module):
    """Single SWSA encoder block."""

    def __init__(self, channels: int, *, hidden_channels: int = 1024, num_heads: int = 8) -> None:
        super().__init__()
        self.swsa = PatchSA(channels, heads=num_heads)
        self.fc1 = nn.Conv2d(channels, hidden_channels, 1)
        self.fc2 = nn.Conv2d(hidden_channels, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.norm1(src + self.swsa(src))
        return self.norm2(src + self.fc2(self.act(self.fc1(src))))


class SWSAIFIEncoder(nn.Module):
    """Encoder wrapper applied to the projected S5 feature map."""

    def __init__(self, dim: int = 256, *, num_heads: int = 8, hidden_channels: int = 1024) -> None:
        super().__init__()
        self.block = TransformerEncoderLayerSWSA(
            dim,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
