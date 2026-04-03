# PRD-05: API & Docker

> Module: DEF-UAVDETR | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Expose UAV-DETR through a stable FastAPI service and a reproducible GPU-capable Docker image suitable for ANIMA composition.

## Context
The paper stops at model evaluation. ANIMA needs a service boundary that other modules can call, plus a container that can run the inference stack without depending on the vendored Ultralytics project layout.

Paper references:
- §3.1: image-in to detection-out contract
- §4.6: overhead matters, so service design must keep the paper model intact and avoid extra latency where possible

## Acceptance Criteria
- [ ] `POST /predict` accepts image upload or URI and returns normalized detection JSON.
- [ ] `GET /healthz` and `GET /readyz` expose service health and checkpoint readiness.
- [ ] Docker build supports CPU smoke tests and a CUDA runtime variant.
- [ ] Compose file mounts the staged checkpoints and datasets through explicit paths.
- [ ] Test: `uv run pytest tests/test_api.py tests/test_container_config.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_def_uavdetr/api/app.py` | FastAPI application | §3.1 | ~180 |
| `src/anima_def_uavdetr/api/schemas.py` | Request/response models | — | ~80 |
| `docker/Dockerfile.api` | API image | — | ~120 |
| `docker-compose.api.yml` | Local stack runner | — | ~80 |
| `.env.example` | Runtime variables | — | ~40 |
| `tests/test_api.py` | API tests | — | ~120 |
| `tests/test_container_config.py` | Docker/compose config tests | — | ~80 |

## Architecture Detail

### Inputs
- `UploadFile` or image URI
- `confidence: float`

### Outputs
- JSON `{"detections": [{"bbox_xyxy": [...], "score": 0.0, "class_id": 0, "label": "uav"}]}`

### Algorithm
```python
app = FastAPI(title="DEF-UAVDETR API")


@app.post("/predict")
async def predict(file: UploadFile, conf: float = 0.25):
    image = await decode_upload(file)
    detections = predictor.predict([image], conf=conf)
    return DetectionResponse.from_tensor(detections[0])
```

## Dependencies
```toml
fastapi = ">=0.112"
uvicorn = ">=0.30"
python-multipart = ">=0.0.9"
```

## Data Requirements
| Asset | Size | Path | Download |
|------|------|------|----------|
| Inference checkpoint | model-dependent | `/Volumes/AIFlowDev/RobotFlowLabs/models/def_uavdetr/checkpoints/uavdetr-best.pt` | Produced by PRD-04 |

## Test Plan
```bash
uv run pytest tests/test_api.py tests/test_container_config.py -v
```

## References
- Paper: §3.1 "Overall Architecture"
- Paper: §4.6 "Algorithm Failures and Limitations"
- Depends on: PRD-03
- Feeds into: PRD-07
