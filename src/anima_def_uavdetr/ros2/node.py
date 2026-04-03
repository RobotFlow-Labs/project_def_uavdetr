"""ROS2 detector node for DEF-UAVDETR.

Subscribes to ``sensor_msgs/Image`` on ``/camera/image_raw``, runs the
UAV-DETR predictor, and publishes detection results.  When ``rclpy`` is
not available the module can still be imported for testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import UavDetrSettings
from ..infer import DefUavDetrPredictor
from .messages import tensor_to_detection_array

logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    """Plain-data config for the detector node (no rclpy dependency)."""

    checkpoint_path: str = ""
    device: str = "auto"
    confidence: float = 0.25
    max_det: int = 300
    frame_skip: int = 0
    image_topic: str = "/camera/image_raw"
    detection_topic: str = "/def_uavdetr/detections"
    debug_image_topic: str = "/def_uavdetr/debug_image"
    image_width: int = 640
    image_height: int = 640


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def build_predictor(
    cfg: NodeConfig, settings: UavDetrSettings | None = None,
) -> DefUavDetrPredictor:
    """Construct a predictor from a NodeConfig."""
    settings = settings or UavDetrSettings()
    device = _resolve_device(cfg.device)
    ckpt = Path(cfg.checkpoint_path) if cfg.checkpoint_path else None
    if ckpt is not None and not ckpt.exists():
        logger.warning("Checkpoint %s does not exist — running with random weights", ckpt)
        ckpt = None
    return DefUavDetrPredictor(checkpoint_path=ckpt, device=device, settings=settings)


class DefUavDetrNodeBase:
    """Logic core of the ROS2 node, decoupled from rclpy for testability."""

    def __init__(self, cfg: NodeConfig | None = None) -> None:
        self.cfg = cfg or NodeConfig()
        self.predictor = build_predictor(self.cfg)
        self._frame_counter = 0

    def on_image(self, image_array: np.ndarray) -> dict | None:
        """Process a single RGB image array and return detection dict or None if skipped."""
        self._frame_counter += 1
        if self.cfg.frame_skip > 0 and (self._frame_counter % (self.cfg.frame_skip + 1)) != 1:
            return None

        results = self.predictor.predict(
            image_array, conf=self.cfg.confidence, max_det=self.cfg.max_det,
        )
        det_array = tensor_to_detection_array(
            results[0],
            image_width=self.cfg.image_width,
            image_height=self.cfg.image_height,
        )
        return {
            "detections": [
                {
                    "bbox": {
                        "center_x": d.bbox.center_x,
                        "center_y": d.bbox.center_y,
                        "size_x": d.bbox.size_x,
                        "size_y": d.bbox.size_y,
                    },
                    "score": d.score,
                    "class_id": d.class_id,
                    "label": d.label,
                }
                for d in det_array.detections
            ],
            "frame_id": det_array.header_frame_id,
        }


def _try_run_ros2_node() -> None:
    """Launch a full ROS2 node if rclpy is available."""
    try:
        import rclpy  # type: ignore[import-untyped]
    except ImportError:
        logger.error("rclpy not available — cannot start ROS2 node")
        return

    rclpy.init()
    logger.info("DEF-UAVDETR ROS2 node started (stub — full node requires anima-base image)")
    rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _try_run_ros2_node()
