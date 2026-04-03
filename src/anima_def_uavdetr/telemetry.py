"""Lightweight telemetry counters for DEF-UAVDETR.

Paper reference: Section 4.6 — overhead and failure tracking.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(eq=False)
class InferenceMetrics:
    """Thread-safe inference timing and failure counters."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    total_inferences: int = 0
    total_detections: int = 0
    total_errors: int = 0
    total_frames_skipped: int = 0
    _latency_sum_ms: float = 0.0
    _latency_max_ms: float = 0.0
    _start_time: float = field(default_factory=time.monotonic)

    def record_inference(self, latency_ms: float, num_detections: int) -> None:
        with self._lock:
            self.total_inferences += 1
            self.total_detections += num_detections
            self._latency_sum_ms += latency_ms
            if latency_ms > self._latency_max_ms:
                self._latency_max_ms = latency_ms

    def record_error(self) -> None:
        with self._lock:
            self.total_errors += 1

    def record_skip(self, count: int = 1) -> None:
        with self._lock:
            self.total_frames_skipped += count

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if self.total_inferences == 0:
                return 0.0
            return self._latency_sum_ms / self.total_inferences

    @property
    def max_latency_ms(self) -> float:
        with self._lock:
            return self._latency_max_ms

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def throughput_fps(self) -> float:
        elapsed = self.uptime_s
        if elapsed <= 0:
            return 0.0
        with self._lock:
            return self.total_inferences / elapsed

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of current metrics.

        All values are read under a single lock acquisition to avoid
        the deadlock that would occur if we called the locking properties.
        """
        elapsed = self.uptime_s
        with self._lock:
            n = self.total_inferences
            avg = (self._latency_sum_ms / n) if n > 0 else 0.0
            fps = (n / elapsed) if elapsed > 0 else 0.0
            return {
                "total_inferences": n,
                "total_detections": self.total_detections,
                "total_errors": self.total_errors,
                "total_frames_skipped": self.total_frames_skipped,
                "avg_latency_ms": round(avg, 2),
                "max_latency_ms": round(self._latency_max_ms, 2),
                "uptime_s": round(elapsed, 2),
                "throughput_fps": round(fps, 2),
            }
