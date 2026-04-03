# DEF-UAVDETR — Execution Ledger

Resume rule: Read this file COMPLETELY before writing any code.
This project covers exactly ONE paper: UAV-DETR: Anti-Drone Detection.

## 1. Working Rules
- Work only inside `project_def_uavdetr/`
- This wave has 17 parallel projects, 17 papers, 17 agents
- Prefix every commit with `[DEF-UAVDETR]`
- Stage only `project_def_uavdetr/` files
- VERIFY THE PAPER BEFORE BUILDING ANYTHING

## 2. The Paper
- **Title**: UAV-DETR: Anti-Drone Detection
- **ArXiv**: 2603.22841
- **Link**: https://arxiv.org/abs/2603.22841
- **Repo**: https://github.com/wd-sir/UAVDETR
- **Compute**: GPU-NEED
- **Verification status**: ArXiv ID ✅ | Repo ✅ | Paper read ✅

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: Full pipeline built. Tensor cache building. Training next.
- **MVP Readiness**: 72%
- **Accomplished**:
  - PRD-01 foundation ✅
  - PRD-02 core model ✅
  - PRD-03 inference pipeline ✅
  - PRD-04 evaluation (BLOCKED — datasets not in paper format)
  - PRD-05 API + Docker ✅
  - PRD-06 ROS2 integration ✅
  - PRD-07 production + release ✅
  - ANIMA infra (anima_module.yaml, serve.py, Dockerfile.serve) ✅
  - Code review: fixed C1 deadlock, H3 target collation, H6 cu128, H5 compose conflict
  - Seraphim dataset extracted: 75K train + 8K test (640x640 YOLO format)
  - Tensor cache build script ready
  - CUDA training pipeline ready (torch.compile, AMP, cosine LR, early stopping)
  - pyproject.toml updated to cu128
- **TODO**:
  1. Wait for tensor cache build to complete
  2. Launch training on GPU 6 with nohup+disown
  3. Monitor VRAM usage (target 60-70%)
  4. After training: export pth → safetensors → ONNX → TRT FP16/FP32
  5. Push to HF: ilessio-aiflowlab/project_def_uavdetr
  6. Git commit + push
- **Blockers**: Tensor cache build in progress

## 4. Datasets
### Seraphim Drone Detection Dataset
| Dataset | Size | Path | Format | Status |
|---------|------|------|--------|--------|
| Seraphim train | 75,134 images | /mnt/forge-data/datasets/uav_detection/seraphim/train/ | YOLO 640x640 | READY |
| Seraphim test | 8,349 images | /mnt/forge-data/datasets/uav_detection/seraphim/test/ | YOLO 640x640 | READY |
| Tensor cache | ~110GB | /mnt/forge-data/shared_infra/datasets/seraphim_*.pt | torch tensors | BUILDING |

## 5. Hardware
- GPU 6 (NVIDIA L4, 23GB VRAM) — reserved for this module
- CUDA 12.8, torch 2.11.0+cu128

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Project scaffolded |
| 2026-04-03 | Codex | PRD-01/02/03 complete |
| 2026-04-03 | Opus 4.6 | PRD-05/06/07 + ANIMA infra + code review fixes. 5 commits. Tensor cache building. |
