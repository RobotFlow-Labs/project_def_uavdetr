# DEF-UAVDETR: UAV-DETR — Implementation PRD
## ANIMA Wave-7 Defense Module

**Status:** PRD-05/06/07 complete, PRD-04 data-blocked, training pipeline ready
**Version:** 0.2
**Date:** 2026-04-03
**Paper:** UAV-DETR: DETR for Anti-Drone Target Detection
**Paper Link:** https://arxiv.org/abs/2603.22841
**Repo:** https://github.com/wd-sir/UAVDETR
**Compute:** GPU-first, paper reproduced on RTX 3090 24 GB
**Functional Name:** DEF-uavdetr
**Stack:** Defense

## 1. Executive Summary
UAV-DETR is a paper-faithful real-time anti-drone detector built on RT-DETR with four paper-specific improvements: a WTConv-enhanced backbone, an SWSA-IFI encoder, an ECFRFN neck, and an Inner-CIoU + NWD hybrid regression loss. The paper reports `62.56 mAP50:95` on a custom UAV dataset and `67.15 mAP50:95` on DUT-ANTI-UAV while cutting parameters to about `11.96M`.

This repo currently contains only a scaffold plus a vendored reference implementation in `repositories/UAVDETR`. The first build step is not model coding; it is cleanup of the stale `INARI` naming and package paths so the module can be implemented as `DEF-UAVDETR` without leaking incorrect metadata.

## 2. Paper Verification Status
- [x] ArXiv ID verified
- [x] Local paper PDF present at `papers/2603.22841_UAV-DETR.pdf`
- [x] Reference repo vendored at `repositories/UAVDETR`
- [x] Paper read completely
- [x] Architecture extracted from paper and repo YAML
- [ ] Datasets staged on shared volume
- [ ] Metrics reproduced locally
- [ ] Final release artifacts generated
- **Verdict:** VALID PAPER, BUILDABLE MODULE

## Build Plan — Executable PRDs

> Total PRDs: 7 | Tasks: 28 | Status: 24/28 complete

| # | PRD | Title | Priority | Tasks | Status |
|---|-----|-------|----------|-------|--------|
| 1 | [PRD-01](prds/PRD-01-foundation.md) | Foundation & Config | P0 | 4 | ✅ |
| 2 | [PRD-02](prds/PRD-02-core-model.md) | Core Model | P0 | 4 | ✅ |
| 3 | [PRD-03](prds/PRD-03-inference.md) | Inference Pipeline | P0 | 4 | ✅ |
| 4 | [PRD-04](prds/PRD-04-evaluation.md) | Evaluation & Reproduction | P1 | 4 | BLOCKED |
| 5 | [PRD-05](prds/PRD-05-api-docker.md) | API & Docker | P1 | 4 | ✅ |
| 6 | [PRD-06](prds/PRD-06-ros2.md) | ROS2 Integration | P1 | 4 | ✅ |
| 7 | [PRD-07](prds/PRD-07-production.md) | Production & Release | P2 | 4 | ✅ |

## 3. What We Take From The Paper
- WTConv backbone stages that preserve high-frequency detail for tiny aerial targets.
- SWSA-IFI encoder operating on the deepest semantic feature map instead of the standard AIFI path.
- ECFRFN with SBA and RepNCSPELAN4 for cross-scale recalibration and fusion.
- Inner-CIoU + NWD hybrid regression objective for small-box stability.
- Paper evaluation targets on both the custom UAV dataset and DUT-ANTI-UAV.

## 4. What We Skip
- Training all 11 comparison baselines inside this repo. We only need the UAV-DETR path plus evaluators that can compare against published numbers.
- Non-paper sensor fusion. The paper is RGB-only; ZED 2i topic integration is fine, but LiDAR fusion is future work.
- Hard dependency on the vendored Ultralytics code tree at runtime. We use it as reference truth, not as the final ANIMA package boundary.

## 5. What We Adapt
- Normalize the package and metadata from stale `INARI` scaffolding to `DEF-UAVDETR`.
- Wrap the model in ANIMA-friendly Python, CLI, API, Docker, and ROS2 surfaces.
- Add explicit release, benchmark, and telemetry tooling not present in the paper.

## 6. Architecture

### Paper-faithful core path
```text
Input RGB frame Tensor[B,3,640,640]
  -> WTConv backbone
     S2 [B, 64,160,160]
     S3 [B,128, 80, 80]
     S4 [B,256, 40, 40]
     S5 [B,512, 20, 20]
  -> project S5 to 256 channels
  -> SWSA-IFI encoder
     F5 [B,256,20,20]
  -> ECFRFN neck
     P2..P5 all 256 channels
  -> RT-DETR decoder with 300 queries
  -> class logits + box predictions
  -> Inner-CIoU + NWD loss during training
```

### Reference code anchors
- YAML graph: `repositories/UAVDETR/ultralytics/cfg/models/UAV-DETR.yaml`
- WTConv implementation: `repositories/UAVDETR/ultralytics/nn/extra_modules/wtconv2d.py`
- SWSA encoder: `repositories/UAVDETR/ultralytics/nn/extra_modules/SWSATransformer.py`
- Neck blocks: `repositories/UAVDETR/ultralytics/nn/extra_modules/block.py`

## 7. Implementation Phases

### Phase 1 — Foundation ✅
- [x] Rename scaffold metadata from `INARI` to `DEF-UAVDETR`
- [x] Create typed config and dataset manifests
- [x] Implement frame sampler and fixture-backed tests

### Phase 2 — Core Reproduction ✅
- [x] Implement WTConv backbone, SWSA-IFI, ECFRFN, decoder wrapper, and hybrid loss
- [x] Verify end-to-end forward pass shapes and training contract

### Phase 3 — Inference ✅ / Evaluation BLOCKED
- [x] Add inference API, CLI, checkpoint loading, and exports
- [ ] Reproduce metrics for custom UAV and DUT-ANTI-UAV
- [ ] Build comparison/failure-case reports

### Phase 4 — ANIMA Integration ⬜
- [ ] FastAPI service and Docker image
- [ ] ROS2 node and launch flow
- [ ] Release manifest, runtime limits, telemetry, and benchmark suite

## 8. Datasets
| Dataset | Size | URL/Source | Phase Needed |
|---------|------|------------|-------------|
| Custom UAV dataset | 14,713 images | Paper §4.1, staged at `/Volumes/AIFlowDev/RobotFlowLabs/datasets/def_uavdetr/uav_dataset/` | Phase 1 |
| DUT-ANTI-UAV | public benchmark | public benchmark, staged at `/Volumes/AIFlowDev/RobotFlowLabs/datasets/def_uavdetr/dut_anti_uav/` | Phase 3 |

## 9. Dependencies on Other Wave Projects
| Needs output from | What it provides |
|------------------|------------------|
| None required | This module is self-contained once datasets are staged |

## 10. Success Criteria
- Match or closely approach the paper’s custom-UAV metrics: `96.82 P`, `94.93 R`, `95.87 F1`, `62.56 mAP50:95`.
- Match or closely approach DUT-ANTI-UAV metrics: `97.09 P`, `95.26 F1`, `67.15 mAP50:95`.
- Preserve the paper-scale model size target of about `11.96M` parameters.
- Export a working inference checkpoint and expose it through both API and ROS2 surfaces.

## 11. Risk Assessment
- The datasets are not staged locally yet.
- The paper and repo disagree on some training defaults, especially `100` vs `300` epochs.
- The paper calls out two hard failure modes: bird-like distractors and urban camouflage.
- The paper also reports a `17.2%` FLOPs overhead over baseline RT-DETR; deployment work must manage that explicitly.

## 12. Task Index
See [tasks/INDEX.md](tasks/INDEX.md) for the full ordered breakdown.

## 13. Near-Term Demo Target
- Minimum credible demo: offline inference + report-backed custom-UAV evaluation.
- Better demo: API service + ROS2 image topic inference with staged checkpoint.
