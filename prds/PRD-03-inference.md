# PRD-03: Inference Pipeline

> Module: DEF-UAVDETR | Priority: P0
> Depends on: PRD-02
> Status: ⬜ Not started

## Objective
Turn the reproduced UAV-DETR network into a usable inference surface with checkpoint loading, image/video prediction, postprocessing, and export hooks.

## Context
The paper emphasizes real-time deployment while the repo exposes only thin scripts. ANIMA needs a stable inference API and export path that preserve the paper architecture instead of forcing users through the vendored Ultralytics tree.

Paper references:
- §3.1: end-to-end pipeline from image to predictions
- §4.4.3: qualitative image predictions
- §4.6: model must eventually tolerate constrained hardware

## Acceptance Criteria
- [ ] A typed checkpoint loader can initialize scratch weights, resumed training checkpoints, and exported inference checkpoints.
- [ ] An inference surface accepts file paths, in-memory tensors, and batched frames.
- [ ] Postprocessing returns `[x1, y1, x2, y2, score, class_id]` detections and optional visualization overlays.
- [ ] CLI entrypoint reproduces the role of `repositories/UAVDETR/detect.py` without hard-coded Windows paths.
- [ ] Export hooks generate at least PyTorch and ONNX outputs, with placeholders for TensorRT and MLX adaptation.
- [ ] Test: `uv run pytest tests/test_infer.py tests/test_checkpoint_io.py tests/test_export.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_def_uavdetr/checkpoints.py` | Checkpoint I/O | §4.2 | ~120 |
| `src/anima_def_uavdetr/postprocess.py` | Query filtering and box decoding | §3.1 | ~120 |
| `src/anima_def_uavdetr/infer.py` | Python inference API | §4.4.3 | ~180 |
| `scripts/run_infer.py` | CLI wrapper | §4.4.3 | ~80 |
| `scripts/export.py` | Export surface | §4.6 | ~100 |
| `tests/test_infer.py` | Inference tests | — | ~120 |
| `tests/test_checkpoint_io.py` | Checkpoint tests | — | ~80 |
| `tests/test_export.py` | Export tests | — | ~80 |

## Architecture Detail

### Inputs
- `images: Tensor[B, 3, 640, 640]`
- `checkpoint_path: Path`
- `conf_threshold: float`

### Outputs
- `detections: list[Tensor[N_i, 6]]`
- `rendered_images: list[np.ndarray]`
- `export_artifacts: dict[str, Path]`

### Algorithm
```python
class DefUavDetrPredictor:
    def __init__(self, model, checkpoint_path):
        self.model = load_checkpoint(model, checkpoint_path)

    @torch.inference_mode()
    def predict(self, images, conf=0.25):
        pred_boxes, pred_logits = self.model(images)
        return postprocess_queries(pred_boxes, pred_logits, conf=conf, max_det=300)
```

## Dependencies
```toml
opencv-python = ">=4.10"
numpy = ">=1.26"
onnx = ">=1.16"
```

## Data Requirements
| Asset | Size | Path | Download |
|------|------|------|----------|
| Trained checkpoint | model-dependent | `/Volumes/AIFlowDev/RobotFlowLabs/models/def_uavdetr/checkpoints/uavdetr-best.pt` | Produced by training |
| Example frames | small sample | `tests/fixtures/` | Create minimal local fixtures |

## Test Plan
```bash
uv run pytest tests/test_infer.py tests/test_checkpoint_io.py tests/test_export.py -v
```

## References
- Paper: §3.1 "Overall Architecture"
- Paper: §4.4.3 "Visual Results"
- Reference impl: `repositories/UAVDETR/detect.py`
- Reference impl: `repositories/UAVDETR/ultralytics/nn/modules/head.py`
- Feeds into: PRD-04, PRD-05, PRD-06, PRD-07
