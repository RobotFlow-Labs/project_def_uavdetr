"""Runtime limits and graceful degradation for DEF-UAVDETR.

Paper reference: Section 4.6 — the paper identifies a 17.2% FLOPs overhead
relative to baseline RT-DETR and documents failure modes around bird-like
distractors and urban camouflage.  This module makes those limits explicit
at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrameSkipPolicy:
    """Adaptive frame-skip: when inference is slower than target, skip frames."""

    target_fps: float = 30.0
    max_skip: int = 4
    _last_inference_ms: float = 0.0

    def recommend_skip(self, inference_ms: float) -> int:
        """Return how many frames to skip given last inference time."""
        self._last_inference_ms = inference_ms
        if inference_ms <= 0:
            return 0
        achievable_fps = 1000.0 / inference_ms
        if achievable_fps >= self.target_fps:
            return 0
        ratio = self.target_fps / max(achievable_fps, 1e-6)
        return min(int(ratio) - 1, self.max_skip)


@dataclass
class GracefulDegradation:
    """CPU fallback and resolution reduction under resource pressure."""

    allow_cpu_fallback: bool = True
    min_image_size: int = 320
    default_image_size: int = 640
    _active_backend: str = "cuda"

    def should_reduce_resolution(self, vram_used_pct: float) -> int:
        """Return recommended image size given VRAM pressure."""
        if vram_used_pct > 90:
            return self.min_image_size
        if vram_used_pct > 80:
            return (self.default_image_size + self.min_image_size) // 2
        return self.default_image_size

    def should_fallback_cpu(self, gpu_available: bool) -> bool:
        """Return True if we should fall back to CPU inference."""
        if gpu_available:
            return False
        return self.allow_cpu_fallback


@dataclass
class RuntimeLimits:
    """Aggregate runtime limits derived from the paper."""

    known_false_positives: list[str] = field(
        default_factory=lambda: ["bird-like distractors", "kites", "balloons"]
    )
    known_false_negatives: list[str] = field(
        default_factory=lambda: ["urban camouflage", "heavy occlusion", "extreme distance"]
    )
    flops_overhead_pct: float = 17.2
    model_params: int = 11_962_040
    max_queries: int = 300
    frame_skip: FrameSkipPolicy = field(default_factory=FrameSkipPolicy)
    degradation: GracefulDegradation = field(default_factory=GracefulDegradation)

    def to_dict(self) -> dict:
        return {
            "known_false_positives": self.known_false_positives,
            "known_false_negatives": self.known_false_negatives,
            "flops_overhead_pct": self.flops_overhead_pct,
            "model_params": self.model_params,
            "max_queries": self.max_queries,
        }
