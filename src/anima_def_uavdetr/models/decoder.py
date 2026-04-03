"""Lightweight RT-DETR-style decoder wrapper."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, num_layers: int) -> None:
        super().__init__()
        layers = []
        dims = [input_dim, *([hidden_dim] * (num_layers - 1)), output_dim]
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RTDETRHead(nn.Module):
    """Multi-scale query decoder that preserves the RT-DETR output contract."""

    def __init__(
        self,
        *,
        num_classes: int = 1,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_heads: int = 8,
        num_decoder_layers: int = 3,
        num_feature_levels: int = 4,
        memory_size: int = 512,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Parameter(torch.zeros(num_feature_levels, hidden_dim))
        self.encoder_score_head = nn.Linear(hidden_dim, num_classes)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="relu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        nn.init.normal_(self.level_embed, std=1.0 / math.sqrt(hidden_dim))

    def _flatten_features(self, pyramid_features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        flattened = []
        for level, feature in enumerate(pyramid_features):
            tokens = feature.flatten(2).transpose(1, 2)
            flattened.append(tokens + self.level_embed[level].view(1, 1, -1))
        return torch.cat(flattened, dim=1)

    def _select_memory(self, memory: torch.Tensor) -> torch.Tensor:
        scores = self.encoder_score_head(memory).amax(dim=-1)
        topk = min(self.memory_size, memory.shape[1])
        indices = scores.topk(topk, dim=1).indices
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        return torch.gather(memory, 1, gather_indices)

    def forward(
        self,
        pyramid_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del targets

        memory = self._flatten_features(pyramid_features)
        memory = self._select_memory(memory)
        batch_size = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        pred_boxes = self.bbox_head(decoded).sigmoid()
        pred_logits = self.class_head(decoded)
        return pred_boxes, pred_logits
