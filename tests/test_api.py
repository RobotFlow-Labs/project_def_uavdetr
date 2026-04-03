"""Tests for the DEF-UAVDETR FastAPI service."""

from __future__ import annotations

import pytest
import torch

from anima_def_uavdetr.api.schemas import (
    Detection,
    DetectionResponse,
    HealthResponse,
    InfoResponse,
    ReadyResponse,
)


class TestSchemas:
    """Schema round-trip and validation."""

    def test_detection_round_trip(self):
        d = Detection(bbox_xyxy=[0.1, 0.2, 0.3, 0.4], score=0.9, class_id=0, label="uav")
        assert d.score == 0.9
        assert len(d.bbox_xyxy) == 4

    def test_detection_response_from_tensor(self):
        tensor = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.95, 0.0]])
        resp = DetectionResponse.from_tensor(tensor, width=640, height=480)
        assert len(resp.detections) == 1
        assert resp.detections[0].score == pytest.approx(0.95, abs=1e-4)
        assert resp.image_width == 640

    def test_detection_response_empty_tensor(self):
        tensor = torch.zeros((0, 6))
        resp = DetectionResponse.from_tensor(tensor)
        assert len(resp.detections) == 0

    def test_health_defaults(self):
        h = HealthResponse()
        assert h.status == "ok"
        assert h.module == "def-uavdetr"

    def test_ready_defaults(self):
        r = ReadyResponse()
        assert r.ready is False
        assert r.module == "def-uavdetr"

    def test_info_defaults(self):
        info = InfoResponse()
        assert info.arxiv == "2603.22841"
        assert info.num_classes == 1
        assert info.image_size == 640


class TestAppImport:
    """Verify app factory can be imported without side-effects."""

    def test_create_app_returns_fastapi(self):
        from anima_def_uavdetr.api.app import create_app

        application = create_app()
        assert application.title == "DEF-UAVDETR API"

    def test_app_routes_registered(self):
        from anima_def_uavdetr.api.app import create_app

        application = create_app()
        paths = {r.path for r in application.routes}
        assert "/health" in paths
        assert "/healthz" in paths
        assert "/ready" in paths
        assert "/readyz" in paths
        assert "/info" in paths
        assert "/predict" in paths
