# DEF-UAVDETR PRD Suite

This directory contains the execution PRDs for reproducing and productizing the UAV-DETR paper inside ANIMA.

## Build Order

| Order | PRD | Title | Priority | Depends On |
|------|-----|-------|----------|------------|
| 1 | [PRD-01](PRD-01-foundation.md) | Foundation & Config | P0 | None |
| 2 | [PRD-02](PRD-02-core-model.md) | Core Model | P0 | PRD-01 |
| 3 | [PRD-03](PRD-03-inference.md) | Inference Pipeline | P0 | PRD-02 |
| 4 | [PRD-04](PRD-04-evaluation.md) | Evaluation & Reproduction | P1 | PRD-02, PRD-03 |
| 5 | [PRD-05](PRD-05-api-docker.md) | API & Docker | P1 | PRD-03 |
| 6 | [PRD-06](PRD-06-ros2.md) | ROS2 Integration | P1 | PRD-03 |
| 7 | [PRD-07](PRD-07-production.md) | Production & Release | P2 | PRD-04, PRD-05, PRD-06 |

## Non-Negotiables

- The implementation must remain paper-faithful for the model path used in PRD-02 and PRD-04.
- All naming must be normalized from the stale `INARI` scaffold to `DEF-UAVDETR`.
- The first complete reproduction target is the paper’s custom UAV dataset table, then DUT-ANTI-UAV.
- The ANIMA surfaces are additive; they must not silently change the paper model topology.
