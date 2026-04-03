"""Tests for the ROS2 integration layer (no rclpy required)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anima_def_uavdetr.ros2.messages import (
    BBox2D,
    Detection2D,
    Detection2DArray,
    tensor_to_detection_array,
)
from anima_def_uavdetr.ros2.node import DefUavDetrNodeBase, NodeConfig, build_predictor


class TestMessages:
    """Test detection message conversion helpers."""

    def test_tensor_to_detection_array_basic(self):
        tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.9, 0.0]])
        result = tensor_to_detection_array(tensor, image_width=640, image_height=480)
        assert isinstance(result, Detection2DArray)
        assert len(result.detections) == 1
        det = result.detections[0]
        assert det.score == pytest.approx(0.9, abs=1e-4)
        assert det.label == "uav"
        assert det.bbox.size_x == pytest.approx((0.3 - 0.1) * 640, abs=1e-2)
        assert det.bbox.size_y == pytest.approx((0.4 - 0.2) * 480, abs=1e-2)

    def test_tensor_to_detection_array_empty(self):
        tensor = torch.zeros((0, 6))
        result = tensor_to_detection_array(tensor)
        assert len(result.detections) == 0

    def test_tensor_to_detection_array_multi(self):
        tensor = torch.tensor([
            [0.1, 0.1, 0.2, 0.2, 0.8, 0.0],
            [0.5, 0.5, 0.7, 0.7, 0.6, 0.0],
        ])
        result = tensor_to_detection_array(tensor)
        assert len(result.detections) == 2

    def test_detection2d_defaults(self):
        d = Detection2D()
        assert d.score == 0.0
        assert d.label == "uav"

    def test_bbox2d_defaults(self):
        b = BBox2D()
        assert b.center_x == 0.0


class TestNodeConfig:
    """Test node configuration dataclass."""

    def test_defaults(self):
        cfg = NodeConfig()
        assert cfg.confidence == 0.25
        assert cfg.frame_skip == 0
        assert cfg.image_topic == "/camera/image_raw"
        assert cfg.detection_topic == "/def_uavdetr/detections"

    def test_override(self):
        cfg = NodeConfig(confidence=0.5, frame_skip=2)
        assert cfg.confidence == 0.5
        assert cfg.frame_skip == 2


class TestNodeBase:
    """Test the ROS2 node logic core without rclpy."""

    def test_on_image_returns_dict(self):
        node = DefUavDetrNodeBase(NodeConfig(device="cpu"))
        fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = node.on_image(fake_image)
        assert result is not None
        assert "detections" in result
        assert isinstance(result["detections"], list)

    def test_frame_skip(self):
        cfg = NodeConfig(device="cpu", frame_skip=1)
        node = DefUavDetrNodeBase(cfg)
        fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        r1 = node.on_image(fake)  # frame 1 → process
        r2 = node.on_image(fake)  # frame 2 → skip
        r3 = node.on_image(fake)  # frame 3 → process
        assert r1 is not None
        assert r2 is None
        assert r3 is not None

    def test_build_predictor_no_checkpoint(self):
        cfg = NodeConfig(device="cpu", checkpoint_path="")
        predictor = build_predictor(cfg)
        assert predictor is not None


class TestLaunchFile:
    """Test launch file can be imported."""

    def test_generate_launch_description_importable(self):
        from anima_def_uavdetr.ros2.launch.uavdetr_launch import generate_launch_description
        # Without ROS2 it returns None — that's fine
        generate_launch_description()  # just verify it doesn't crash
