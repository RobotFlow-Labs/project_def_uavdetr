"""Detection conversion helpers for ROS2 message types.

Converts between the internal ``[x1, y1, x2, y2, score, class_id]`` tensor
format and ``vision_msgs/Detection2DArray``.  When ``vision_msgs`` is not
importable (no ROS2 environment) the helpers return plain dicts so that unit
tests can still exercise the conversion logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BBox2D:
    """Lightweight mirror of vision_msgs/BoundingBox2D."""

    center_x: float = 0.0
    center_y: float = 0.0
    size_x: float = 0.0
    size_y: float = 0.0


@dataclass
class Detection2D:
    """Lightweight mirror of vision_msgs/Detection2D."""

    bbox: BBox2D = field(default_factory=BBox2D)
    score: float = 0.0
    class_id: int = 0
    label: str = "uav"


@dataclass
class Detection2DArray:
    """Lightweight mirror of vision_msgs/Detection2DArray."""

    header_stamp_sec: int = 0
    header_stamp_nanosec: int = 0
    header_frame_id: str = ""
    detections: list[Detection2D] = field(default_factory=list)


def tensor_to_detection_array(
    detections: torch.Tensor,
    *,
    frame_id: str = "camera",
    stamp_sec: int = 0,
    stamp_nanosec: int = 0,
    image_width: int = 640,
    image_height: int = 640,
) -> Detection2DArray:
    """Convert ``[N, 6]`` tensor to a ``Detection2DArray`` dataclass.

    The input tensor has rows ``[x1, y1, x2, y2, score, class_id]`` in
    normalised coordinates.  The output bounding boxes are in pixel
    coordinates.
    """
    items: list[Detection2D] = []
    if detections.numel() > 0:
        for row in detections.tolist():
            x1, y1, x2, y2, score, cls_id = row
            cx = (x1 + x2) / 2.0 * image_width
            cy = (y1 + y2) / 2.0 * image_height
            w = (x2 - x1) * image_width
            h = (y2 - y1) * image_height
            items.append(
                Detection2D(
                    bbox=BBox2D(center_x=cx, center_y=cy, size_x=w, size_y=h),
                    score=score,
                    class_id=int(cls_id),
                    label="uav",
                )
            )
    return Detection2DArray(
        header_stamp_sec=stamp_sec,
        header_stamp_nanosec=stamp_nanosec,
        header_frame_id=frame_id,
        detections=items,
    )


def try_import_ros2_msgs() -> dict[str, Any] | None:
    """Attempt to import real ROS2 message types; return None if unavailable."""
    try:
        from vision_msgs.msg import (  # type: ignore[import-untyped]
            Detection2D as RosDetection2D,
        )
        from vision_msgs.msg import (
            Detection2DArray as RosDetection2DArray,
        )

        return {
            "Detection2D": RosDetection2D,
            "Detection2DArray": RosDetection2DArray,
        }
    except ImportError:
        return None
