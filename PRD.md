# INARI: UAV-DETR: Anti-Drone Detection — Implementation PRD
## ANIMA Wave-7 Module #16

**Status:** Scaffold
**Version:** 0.1
**Date:** 2026-04-03
**Paper:** UAV-DETR: Anti-Drone Detection
**Paper Link:** https://arxiv.org/abs/2603.22841
**Repo:** https://github.com/wd-sir/UAVDETR
**Compute:** GPU-NEED
**Functional Name:** DEF-uavdetr
**Stack:** Defense

## 1. Executive Summary
(TODO — fill after reading paper)

## 2. Paper Verification Status
- [x] ArXiv ID verified (Kill Protocol V2)
- [x] GitHub repo confirmed accessible
- [ ] Paper read completely
- [ ] Reference repo cloned and tested
- [ ] Datasets confirmed accessible
- [ ] Metrics appear reproducible
- [ ] No red flags found
- **Verdict:** PENDING

## 3. What We Take From The Paper
(TODO)

## 4. What We Skip
(TODO)

## 5. What We Adapt
(TODO)

## 6. Architecture
(TODO — module design, inputs, outputs, interfaces)

## 7. Implementation Phases

### Phase 1 — Scaffold + Verification ⬜
- [x] Project structure created
- [ ] Reference repo cloned to /Volumes/AIFlowDev/RobotFlowLabs/repos/wave7
- [ ] Demo/inference tested on reference data
- [ ] Paper claims verified on their benchmark

### Phase 2 — Reproduce on Reference Dataset ⬜
- [ ] Core method implemented
- [ ] Train/eval on paper's dataset
- [ ] Match paper metrics (within ±5%)

### Phase 3 — Adapt to Our Hardware ⬜
- [ ] Data pipeline for our sensors (ZED 2i, Unitree L2)
- [ ] MLX port (if GPU-NEED == GPU-NEED, at least inference)
- [ ] Real sensor inference tests

### Phase 4 — ANIMA Integration ⬜
- [ ] ROS2 bridge node
- [ ] Docker container
- [ ] API endpoints for stack composition

## 8. Datasets
| Dataset | Size | URL | Phase Needed |
|---------|------|-----|-------------|
| (TODO) | — | — | Phase 1 |

## 9. Dependencies on Other Wave Projects
| Needs output from | What it provides |
|------------------|------------------|
| (TODO or None) | — |

## 10. Success Criteria
(TODO — quantitative benchmarks from paper)

## 11. Risk Assessment
(TODO — what could make this paper useless for us)

## 12. Build Plan
| PRD# | Task | Status |
|------|------|--------|
| PRD-1 | Scaffold + Verification | ⬜ |
| PRD-2 | Reproduce | ⬜ |
| PRD-3 | Adapt to HW | ⬜ |
| PRD-4 | ANIMA Integration | ⬜ |

## 13. Shenzhen Demo (Apr 23-24)
- **Demo-ready target**: Phase 2 minimum, Phase 3 preferred
- **Demo plan**: (TODO)
