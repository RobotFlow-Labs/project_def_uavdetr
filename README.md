# DEF-UAVDETR — ANIMA Module

> **UAV-DETR: Anti-Drone Detection**
> Paper: [arXiv:2603.22841](https://arxiv.org/abs/2603.22841)

Part of the [ANIMA Intelligence Compiler Suite](https://github.com/RobotFlow-Labs) by AIFLOW LABS LIMITED.

## Domain
Defense

## Status
- [ ] Paper read + ASSETS.md created
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

## License
MIT — AIFLOW LABS LIMITED
