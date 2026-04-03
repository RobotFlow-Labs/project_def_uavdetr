# PRD-07: Production & Release

> Module: DEF-UAVDETR | Priority: P2
> Depends on: PRD-04, PRD-05, PRD-06
> Status: ⬜ Not started

## Objective
Prepare UAV-DETR for production release with export validation, observability, graceful degradation, and explicit release gates tied to the paper’s measured limits.

## Context
The paper’s discussion section identifies the remaining risks clearly: bird-like distractors, heavy camouflage, and a `17.2%` FLOPs overhead relative to baseline RT-DETR. Productionization must expose those limits rather than hide them.

Paper references:
- §4.6: failure modes and overhead
- §5: future work includes pruning and quantization for stricter hardware envelopes

## Acceptance Criteria
- [ ] Export validation covers PyTorch, ONNX, and at least one placeholder benchmark path for TensorRT or MLX.
- [ ] Release manifest records the measured custom-UAV and DUT metrics and the model size.
- [ ] Service and ROS2 surfaces emit lightweight timing and failure counters.
- [ ] Graceful degradation path supports CPU fallback and configurable frame skipping.
- [ ] Test: `uv run pytest tests/test_release_manifest.py tests/test_runtime_limits.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `scripts/benchmark.py` | Runtime benchmark harness | §4.6 | ~140 |
| `scripts/release.py` | Release bundle generator | §5 | ~120 |
| `src/anima_def_uavdetr/runtime_limits.py` | Fallback and frame-skip policy | §4.6 | ~80 |
| `src/anima_def_uavdetr/telemetry.py` | Latency and failure counters | §4.6 | ~80 |
| `tests/test_release_manifest.py` | Release metadata tests | — | ~100 |
| `tests/test_runtime_limits.py` | Runtime-policy tests | — | ~80 |

## Architecture Detail

### Inputs
- `benchmark_frames: list[Path]`
- `export_artifacts: dict[str, Path]`
- `evaluation_summary: Path`

### Outputs
- `release_manifest.json`
- `benchmark_summary.json`
- `hf_bundle/` or equivalent release directory

### Algorithm
```python
def build_release_manifest(metrics, artifacts):
    return {
        "model": "def-uavdetr",
        "paper_targets": {"custom_map50_95": 62.56, "dut_map50_95": 67.15},
        "artifacts": artifacts,
        "limits": {
            "known_false_positives": ["bird-like distractors"],
            "known_false_negatives": ["urban camouflage"],
        },
    }
```

## Dependencies
```toml
orjson = ">=3.10"
psutil = ">=6.0"
```

## Data Requirements
| Asset | Size | Path | Download |
|------|------|------|----------|
| Final checkpoint | model-dependent | `/Volumes/AIFlowDev/RobotFlowLabs/models/def_uavdetr/checkpoints/uavdetr-best.pt` | Produced by PRD-04 |
| Evaluation summary | markdown/json | `reports/def_uavdetr/` | Produced by PRD-04 |

## Test Plan
```bash
uv run pytest tests/test_release_manifest.py tests/test_runtime_limits.py -v
```

## References
- Paper: §4.6 "Discussion on Algorithm Failures and Limitations"
- Paper: §5 "Conclusion"
- Depends on: PRD-04, PRD-05, PRD-06
