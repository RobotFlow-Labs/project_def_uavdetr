"""Checkpoint I/O for DEF-UAVDETR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def unwrap_state_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize common checkpoint formats to a plain state dict."""

    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load a checkpoint into a model and return the model."""

    checkpoint = torch.load(Path(path), map_location=map_location)
    state_dict = unwrap_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return model


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save model weights in a resumable checkpoint container."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict()}
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)
    return path
