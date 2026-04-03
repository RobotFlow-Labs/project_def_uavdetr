"""Tests for the release manifest builder and telemetry."""

from __future__ import annotations

import pytest

from anima_def_uavdetr.telemetry import InferenceMetrics
from anima_def_uavdetr.version import __version__


class TestInferenceMetrics:
    def test_empty_snapshot(self):
        m = InferenceMetrics()
        snap = m.snapshot()
        assert snap["total_inferences"] == 0
        assert snap["avg_latency_ms"] == 0.0

    def test_record_inference(self):
        m = InferenceMetrics()
        m.record_inference(10.0, 3)
        m.record_inference(20.0, 5)
        assert m.total_inferences == 2
        assert m.total_detections == 8
        assert m.avg_latency_ms == pytest.approx(15.0, abs=0.1)
        assert m.max_latency_ms == pytest.approx(20.0, abs=0.1)

    def test_record_error(self):
        m = InferenceMetrics()
        m.record_error()
        m.record_error()
        assert m.total_errors == 2

    def test_record_skip(self):
        m = InferenceMetrics()
        m.record_skip(3)
        assert m.total_frames_skipped == 3

    def test_throughput(self):
        m = InferenceMetrics()
        m.record_inference(10.0, 1)
        # Just check it returns a positive number
        assert m.throughput_fps >= 0.0

    def test_snapshot_keys(self):
        m = InferenceMetrics()
        snap = m.snapshot()
        expected_keys = {
            "total_inferences",
            "total_detections",
            "total_errors",
            "total_frames_skipped",
            "avg_latency_ms",
            "max_latency_ms",
            "uptime_s",
            "throughput_fps",
        }
        assert set(snap.keys()) == expected_keys


class TestReleaseManifest:
    def test_build_manifest(self):
        from scripts.release import build_release_manifest

        manifest = build_release_manifest()
        assert manifest["model"] == "def-uavdetr"
        assert manifest["arxiv"] == "2603.22841"
        assert manifest["version"] == __version__
        assert "artifacts" in manifest
        assert "limits" in manifest

    def test_manifest_with_metrics(self):
        from scripts.release import build_release_manifest

        metrics = {"custom_uav_map50_95": 61.5}
        manifest = build_release_manifest(metrics=metrics)
        assert manifest["achieved_metrics"]["custom_uav_map50_95"] == 61.5

    def test_manifest_paper_targets(self):
        from scripts.release import build_release_manifest

        manifest = build_release_manifest()
        assert manifest["paper_targets"]["custom_uav_map50_95"] == 62.56
        assert manifest["paper_targets"]["dut_anti_uav_map50_95"] == 67.15
