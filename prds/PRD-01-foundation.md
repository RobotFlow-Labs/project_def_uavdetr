# PRD-01: Foundation & Config

> Module: DEF-UAVDETR | Priority: P0
> Depends on: None
> Status: ✅ Complete

## Objective
Create a clean, correctly named `DEF-UAVDETR` Python package, dataset/config contract, and validation scaffolding that all later paper-faithful model work can build on.

## Context
The current repo is only a scaffold and still points to the stale `INARI` codename in `pyproject.toml`, `configs/default.toml`, `NEXT_STEPS.md`, and `src/anima_inari/`. The paper-specific work cannot proceed safely until the package, config, dataset paths, and test harness are normalized.

Paper references:
- §3.1: `"one frame is randomly selected from every five"`
- §4.1: custom dataset with `14,713` images and a `7:2:1` split
- §4.2: training/evaluation occurs at `640` resolution on a CUDA stack

## Acceptance Criteria
- [x] `src/anima_def_uavdetr/` exists and replaces the stale `src/anima_inari/` package in project metadata.
- [x] Pydantic settings cover dataset roots, split names, training defaults, and export paths.
- [x] Dataset manifest logic validates the custom UAV layout and DUT-ANTI-UAV layout without needing the full training stack.
- [x] Frame-subsampling utility supports the paper’s 1-in-5 training rule.
- [x] Test: `uv run pytest tests/test_config.py tests/test_dataset_manifest.py -v` passes.
- [x] Lint: `uv run ruff check src/ tests/` passes for the renamed package and new config code.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_def_uavdetr/__init__.py` | Correct package entrypoint and version export | — | ~10 |
| `src/anima_def_uavdetr/version.py` | Package version | — | ~5 |
| `src/anima_def_uavdetr/config.py` | Pydantic settings for data, model, export, and runtime | §4.1, §4.2 | ~120 |
| `src/anima_def_uavdetr/data/layout.py` | Dataset path and split manifest helpers | §4.1 | ~120 |
| `src/anima_def_uavdetr/data/sampler.py` | 1-in-5 temporal frame sampler | §3.1, §4.1 | ~80 |
| `tests/test_config.py` | Settings tests | — | ~80 |
| `tests/test_dataset_manifest.py` | Dataset/sampler tests | — | ~120 |
| `configs/default.toml` | Correct codename and package paths | — | ~40 |

## Architecture Detail

### Inputs
- `project_root: Path`
- `dataset_root: Path`
- `sequence_frames: list[Path]`

### Outputs
- `settings: UavDetrSettings`
- `layout: DatasetLayout`
- `sampled_frames: list[Path]`

### Algorithm
```python
# Paper §4.1 + repo scaffolding normalization
from pathlib import Path
from pydantic_settings import BaseSettings


class UavDetrSettings(BaseSettings):
    model_name: str = "def-uavdetr"
    package_name: str = "anima_def_uavdetr"
    image_size: int = 640
    frame_sample_stride: int = 5
    uav_dataset_root: Path
    dut_dataset_root: Path


def sample_training_frames(frames: list[Path], stride: int = 5) -> list[Path]:
    # Paper says one frame is randomly selected from every five adjacent frames.
    return [frames[i] for i in range(0, len(frames), stride)]
```

## Dependencies
```toml
pydantic = ">=2.7"
pydantic-settings = ">=2.2"
pyyaml = ">=6.0"
```

## Data Requirements
| Asset | Size | Path | Download |
|------|------|------|----------|
| Custom UAV dataset root | 14,713 images | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/def_uavdetr/uav_dataset/` | Acquire and stage before PRD-04 |
| DUT-ANTI-UAV root | public benchmark | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/def_uavdetr/dut_anti_uav/` | Acquire and stage before PRD-04 |

## Test Plan
```bash
uv run pytest tests/test_config.py tests/test_dataset_manifest.py -v
uv run ruff check src/ tests/
```

## References
- Paper: §3.1 "Overall Architecture"
- Paper: §4.1 "Dataset Preparation"
- Paper: §4.2 "Implementation Details and Experimental Setup"
- Reference impl: `repositories/UAVDETR/datasets/uav_dataset/data.yaml`
- Feeds into: PRD-02, PRD-03, PRD-04
