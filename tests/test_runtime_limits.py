"""Tests for runtime limits and graceful degradation."""

from __future__ import annotations

from anima_def_uavdetr.runtime_limits import (
    FrameSkipPolicy,
    GracefulDegradation,
    RuntimeLimits,
)


class TestFrameSkipPolicy:
    def test_no_skip_when_fast(self):
        p = FrameSkipPolicy(target_fps=30.0)
        assert p.recommend_skip(10.0) == 0  # 100 FPS > 30

    def test_skip_when_slow(self):
        p = FrameSkipPolicy(target_fps=30.0)
        skip = p.recommend_skip(100.0)  # 10 FPS → need 3x → skip 2
        assert skip == 2

    def test_max_skip_capped(self):
        p = FrameSkipPolicy(target_fps=30.0, max_skip=2)
        skip = p.recommend_skip(500.0)  # 2 FPS → need 15x → capped at 2
        assert skip == 2

    def test_zero_latency(self):
        p = FrameSkipPolicy()
        assert p.recommend_skip(0.0) == 0


class TestGracefulDegradation:
    def test_no_reduce_under_80(self):
        g = GracefulDegradation()
        assert g.should_reduce_resolution(70.0) == 640

    def test_reduce_above_80(self):
        g = GracefulDegradation()
        size = g.should_reduce_resolution(85.0)
        assert size < 640
        assert size > 320

    def test_min_size_above_90(self):
        g = GracefulDegradation()
        assert g.should_reduce_resolution(95.0) == 320

    def test_cpu_fallback_when_no_gpu(self):
        g = GracefulDegradation(allow_cpu_fallback=True)
        assert g.should_fallback_cpu(gpu_available=False) is True

    def test_no_fallback_when_gpu_available(self):
        g = GracefulDegradation()
        assert g.should_fallback_cpu(gpu_available=True) is False


class TestRuntimeLimits:
    def test_defaults(self):
        limits = RuntimeLimits()
        assert limits.flops_overhead_pct == 17.2
        assert limits.model_params == 11_962_040
        assert "bird-like distractors" in limits.known_false_positives

    def test_to_dict(self):
        limits = RuntimeLimits()
        d = limits.to_dict()
        assert "known_false_positives" in d
        assert "known_false_negatives" in d
        assert d["model_params"] == 11_962_040
