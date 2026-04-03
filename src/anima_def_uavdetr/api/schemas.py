"""Request / response models for the DEF-UAVDETR API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single detected UAV bounding box."""

    bbox_xyxy: list[float] = Field(
        ..., min_length=4, max_length=4, description="[x1, y1, x2, y2] normalised"
    )
    score: float = Field(..., ge=0.0, le=1.0)
    class_id: int = Field(default=0, ge=0)
    label: str = "uav"


class DetectionResponse(BaseModel):
    """Batch of detections for a single image."""

    detections: list[Detection] = Field(default_factory=list)
    image_width: int | None = None
    image_height: int | None = None

    @classmethod
    def from_tensor(cls, tensor, *, width: int | None = None, height: int | None = None):
        """Build from a postprocess_queries output tensor ``[N, 6]``."""
        dets: list[Detection] = []
        if tensor is not None and tensor.numel() > 0:
            for row in tensor.tolist():
                dets.append(
                    Detection(
                        bbox_xyxy=row[:4],
                        score=row[4],
                        class_id=int(row[5]),
                        label="uav",
                    )
                )
        return cls(detections=dets, image_width=width, image_height=height)


class HealthResponse(BaseModel):
    """Service health payload."""

    status: str = "ok"
    module: str = "def-uavdetr"
    version: str = ""
    uptime_s: float = 0.0
    gpu_available: bool = False


class ReadyResponse(BaseModel):
    """Service readiness payload."""

    ready: bool = False
    module: str = "def-uavdetr"
    version: str = ""
    weights_loaded: bool = False


class InfoResponse(BaseModel):
    """Extended module info."""

    module: str = "def-uavdetr"
    version: str = ""
    paper: str = "UAV-DETR: Anti-Drone Detection"
    arxiv: str = "2603.22841"
    num_classes: int = 1
    class_names: list[str] = ["uav"]
    image_size: int = 640
    num_queries: int = 300
    backend: str = "cpu"
