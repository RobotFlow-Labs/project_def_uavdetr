"""FastAPI application for DEF-UAVDETR inference."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from ..config import UavDetrSettings, get_settings
from ..infer import DefUavDetrPredictor
from ..version import __version__
from .schemas import (
    DetectionResponse,
    HealthResponse,
    InfoResponse,
    ReadyResponse,
)

_predictor: DefUavDetrPredictor | None = None
_start_time: float = 0.0


def _resolve_device(settings: UavDetrSettings) -> str:
    backend = settings.runtime_backend
    if backend == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return backend


def _load_predictor(settings: UavDetrSettings) -> DefUavDetrPredictor:
    device = _resolve_device(settings)
    ckpt = settings.best_checkpoint_path if settings.best_checkpoint_path.exists() else None
    return DefUavDetrPredictor(
        checkpoint_path=ckpt,
        device=device,
        settings=settings,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _start_time  # noqa: PLW0603
    _start_time = time.monotonic()
    settings = get_settings()
    _predictor = _load_predictor(settings)
    yield
    _predictor = None


app = FastAPI(
    title="DEF-UAVDETR API",
    description="Anti-drone detection service — UAV-DETR paper reproduction",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        module="def-uavdetr",
        version=__version__,
        uptime_s=round(time.monotonic() - _start_time, 2),
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    return await health()


@app.get("/ready", response_model=ReadyResponse)
async def ready():
    loaded = _predictor is not None
    resp = ReadyResponse(
        ready=loaded,
        module="def-uavdetr",
        version=__version__,
        weights_loaded=loaded,
    )
    if not loaded:
        return JSONResponse(content=resp.model_dump(), status_code=503)
    return resp


@app.get("/readyz", response_model=ReadyResponse)
async def readyz():
    return await ready()


@app.get("/info", response_model=InfoResponse)
async def info():
    settings = get_settings()
    return InfoResponse(
        module="def-uavdetr",
        version=__version__,
        image_size=settings.image_size,
        backend=_resolve_device(settings),
    )


@app.post("/predict", response_model=DetectionResponse)
async def predict(
    file: UploadFile = File(...),  # noqa: B008
    conf: float = Query(0.25, ge=0.0, le=1.0),
    max_det: int = Query(300, ge=1, le=3000),
):
    if _predictor is None:
        return JSONResponse(content={"error": "model not loaded"}, status_code=503)

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(content={"error": "unable to decode image"}, status_code=400)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    results = _predictor.predict(rgb, conf=conf, max_det=max_det)
    return DetectionResponse.from_tensor(results[0], width=w, height=h)


def create_app() -> FastAPI:
    """Factory for programmatic use and testing."""
    return app
