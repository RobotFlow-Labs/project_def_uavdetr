"""ANIMA serving node for DEF-UAVDETR.

Implements the AnimaNode interface pattern: ``setup_inference`` loads the
model and ``process`` runs detection on a single input.  Falls back to
standalone FastAPI when ``anima_serve`` is not installed.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .config import UavDetrSettings, get_settings
from .infer import DefUavDetrPredictor
from .version import __version__

logger = logging.getLogger(__name__)


class DefUavDetrServeNode:
    """AnimaNode-compatible serving wrapper for DEF-UAVDETR."""

    def __init__(self, settings: UavDetrSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.predictor: DefUavDetrPredictor | None = None
        self._device = "cpu"

    def setup_inference(self) -> None:
        """Load model weights and configure backend."""
        backend = self.settings.runtime_backend
        if backend == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = backend

        ckpt = self.settings.best_checkpoint_path
        ckpt_path = ckpt if ckpt.exists() else None

        logger.info(
            "Loading DEF-UAVDETR on %s (checkpoint=%s)",
            self._device,
            ckpt_path or "random_init",
        )
        self.predictor = DefUavDetrPredictor(
            checkpoint_path=ckpt_path,
            device=self._device,
            settings=self.settings,
        )
        logger.info("DEF-UAVDETR ready on %s", self._device)

    def process(self, image: np.ndarray, *, conf: float = 0.25) -> list[dict]:
        """Run inference on a single RGB image array.

        Returns a list of detection dicts with keys:
        ``bbox_xyxy``, ``score``, ``class_id``, ``label``.
        """
        if self.predictor is None:
            self.setup_inference()
        assert self.predictor is not None

        results = self.predictor.predict(image, conf=conf)
        tensor = results[0]
        detections = []
        if tensor.numel() > 0:
            for row in tensor.tolist():
                detections.append({
                    "bbox_xyxy": row[:4],
                    "score": row[4],
                    "class_id": int(row[5]),
                    "label": "uav",
                })
        return detections

    def get_status(self) -> dict:
        """Module-specific status fields."""
        return {
            "model_loaded": self.predictor is not None,
            "device": self._device,
            "version": __version__,
        }
