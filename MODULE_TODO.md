# INARI — Design & Implementation Checklist

## Paper: UAV-DETR: Anti-Drone Detection
## ArXiv: 2603.22841
## Repo: https://github.com/wd-sir/UAVDETR

---

## Phase 1: Scaffold + Verification
- [x] Project structure created
- [ ] Paper PDF downloaded to papers/
- [ ] Paper read and annotated
- [ ] Reference repo cloned
- [ ] Reference demo runs successfully
- [ ] Datasets identified and accessibility confirmed
- [ ] CLAUDE.md filled with paper-specific details
- [ ] PRD.md filled with architecture and plan

## Phase 2: Reproduce
- [ ] Core model implemented in src/anima_inari/
- [ ] Training pipeline (scripts/train.py)
- [ ] Evaluation pipeline (scripts/eval.py)
- [ ] Metrics match paper (within ±5%)
- [ ] Dual-compute verified (MLX + CUDA)

## Phase 3: Adapt to Hardware
- [ ] ZED 2i data pipeline (if applicable)
- [ ] Unitree L2 LiDAR pipeline (if applicable)
- [ ] xArm 6 integration (if manipulation module)
- [ ] Real sensor inference test
- [ ] MLX inference port validated

## Phase 4: ANIMA Integration
- [ ] ROS2 bridge node
- [ ] Docker container builds and runs
- [ ] API endpoints defined
- [ ] Integration test with stack: Defense

## Shenzhen Demo Readiness
- [ ] Demo script works end-to-end
- [ ] Demo data prepared
- [ ] Demo runs in < 30 seconds
- [ ] Demo visuals are compelling
