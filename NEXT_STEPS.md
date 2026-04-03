# INARI — Execution Ledger

Resume rule: Read this file COMPLETELY before writing any code.
This project covers exactly ONE paper: UAV-DETR: Anti-Drone Detection.

## 1. Working Rules
- Work only inside `project_inari/`
- This wave has 17 parallel projects, 17 papers, 17 agents
- Prefix every commit with `[INARI]`
- Stage only `project_inari/` files
- VERIFY THE PAPER BEFORE BUILDING ANYTHING

## 2. The Paper
- **Title**: UAV-DETR: Anti-Drone Detection
- **ArXiv**: 2603.22841
- **Link**: https://arxiv.org/abs/2603.22841
- **Repo**: https://github.com/wd-sir/UAVDETR
- **Compute**: GPU-NEED
- **Verification status**: ArXiv ID ✅ | Repo ✅ | Paper read ⬜

## 3. Current Status
- **Date**: 2026-04-03
- **Phase**: Scaffold (just created)
- **MVP Readiness**: 5%
- **Accomplished**: Project scaffolded with standard structure
- **TODO**:
  1. Download paper PDF
  2. Clone reference repo to /Volumes/AIFlowDev/RobotFlowLabs/repos/wave7
  3. Read paper thoroughly
  4. Fill in CLAUDE.md core method / what we take / skip / adapt
  5. Fill in PRD.md sections 1, 3-6, 8-11
  6. Run reference demo/inference
  7. Begin Phase 1 verification
- **Blockers**: None

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
