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
- **Phase**: PRD-04 evaluation blocked by missing datasets; PRD-05 API is the next independent build slice
- **MVP Readiness**: 32%
- **Accomplished**: Planning suite committed; reference repo vendored; paper alignment complete; PRD-01 foundation, PRD-02 core model, and PRD-03 inference pipeline implemented and validated on Python 3.11
- **TODO**:
  1. Stage the custom UAV and DUT-ANTI-UAV datasets on the shared volume so PRD-04 can run
  2. Implement PRD-0401 metrics core once datasets are available
  3. Build PRD-0501 FastAPI schemas/app as the next independent PRD if evaluation remains blocked
  4. Add missing infrastructure files required by autopilot gate checks (`anima_module.yaml`, `Dockerfile.serve`, `docker-compose.serve.yml`, `serve.py`)
  5. Validate Linux/CUDA environment on the training server after prebuild
- **Blockers**: Datasets and production checkpoints are not staged yet

## 4. Datasets
### Required for this paper
| Dataset | Size | URL | Format | Phase Needed |
|---------|------|-----|--------|-------------|
| (TODO after reading paper) | — | — | — | Phase 1 |

### Check shared volume first
/Volumes/AIFlowDev/RobotFlowLabs/datasets

### Download
`bash scripts/download_data.sh`

## 5. Hardware
- ZED 2i stereo camera: Available
- Unitree L2 3D LiDAR: Available
- xArm 6 cobot: Pending purchase
- Mac Studio M-series: MLX dev
- 8x RTX 6000 Pro Blackwell: GCloud

## 6. Session Log
| Date | Agent | What Happened |
|------|-------|---------------|
| 2026-04-03 | ANIMA Research Agent | Project scaffolded |
| 2026-04-03 | Codex | Autopilot recovery completed; starting PRD-01 rename/config/data implementation on Python 3.11 with UV and Linux/CUDA-ready torch sources |
| 2026-04-03 | Codex | PRD-01 complete: renamed package, added typed settings + dataset manifest + frame sampler, pinned repo to Python 3.11 with UV, and passed `uv run pytest tests/test_config.py tests/test_dataset_manifest.py -v` plus `uv run ruff check src/ tests/` |
| 2026-04-03 | Codex | PRD-02 complete: implemented WTConv backbone, SWSA encoder, ECFRFN neck, RT-DETR-style decoder wrapper, and hybrid Inner-CIoU + NWD loss; passed `uv run pytest tests/test_backbone.py tests/test_encoder.py tests/test_neck.py tests/test_model.py tests/test_losses.py -v` |
| 2026-04-03 | Codex | PRD-03 complete: added checkpoint I/O, predictor surface, query postprocess, CLI inference, and export hooks; passed `uv run pytest tests/test_infer.py tests/test_checkpoint_io.py tests/test_export.py -v`, `uv run python scripts/run_infer.py --help`, and full `uv run pytest -v` |
