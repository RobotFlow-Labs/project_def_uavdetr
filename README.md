# DEF-UAVDETR — ANIMA Module

> **UAV-DETR: Anti-Drone Detection**
> Paper: [arXiv:2603.22841](https://arxiv.org/abs/2603.22841)

Part of the [ANIMA Intelligence Compiler Suite](https://github.com/RobotFlow-Labs) by AIFLOW LABS LIMITED.

## Domain
Defense

## Status
- [x] Paper read + ASSETS.md created
- [ ] PRD-01 through PRD-07
- [ ] Training pipeline
- [ ] GPU training
- [ ] Export: pth + safetensors + ONNX + TRT fp16 + TRT fp32
- [ ] Push to HuggingFace
- [ ] Docker serving

## Quick Start
```bash
cd project_def_uavdetr
uv venv .venv --python python3.11 && uv sync
uv run pytest tests/ -v
```

## Build Notes
- Local development target: macOS with Python 3.11 and UV.
- Training stack target: Linux + CUDA. `uv sync` uses the default PyPI wheels on macOS and switches `torch` / `torchvision` to the `cu124` index automatically on Linux.
- Reference implementation for architecture parity lives in [repositories/UAVDETR](/Users/ilessio/Development/AIFLOWLABS/projects/anima-modules/april_03_2026/wave-7/project_def_uavdetr/repositories/UAVDETR).

## License
MIT — AIFLOW LABS LIMITED
