# DEF-UAVDETR — Task Index

## Build Order

| Order | Task | Title | Depends On | PRD |
|------|------|-------|------------|-----|
| 1 | [PRD-0101](PRD-0101.md) | Normalize Package and Metadata | None | PRD-01 |
| 2 | [PRD-0102](PRD-0102.md) | Implement Typed Settings | PRD-0101 | PRD-01 |
| 3 | [PRD-0103](PRD-0103.md) | Build Dataset Layout and Sampler | PRD-0102 | PRD-01 |
| 4 | [PRD-0104](PRD-0104.md) | Add Foundation Tests | PRD-0103 | PRD-01 |
| 5 | [PRD-0201](PRD-0201.md) | Implement WTConv Backbone | PRD-0104 | PRD-02 |
| 6 | [PRD-0202](PRD-0202.md) | Implement SWSA-IFI Encoder | PRD-0201 | PRD-02 |
| 7 | [PRD-0203](PRD-0203.md) | Implement ECFRFN Neck | PRD-0202 | PRD-02 |
| 8 | [PRD-0204](PRD-0204.md) | Integrate Decoder and Hybrid Loss | PRD-0203 | PRD-02 |
| 9 | [PRD-0301](PRD-0301.md) | Add Checkpoint I/O | PRD-0204 | PRD-03 |
| 10 | [PRD-0302](PRD-0302.md) | Add Postprocess and Predictor | PRD-0301 | PRD-03 |
| 11 | [PRD-0303](PRD-0303.md) | Add CLI Inference | PRD-0302 | PRD-03 |
| 12 | [PRD-0304](PRD-0304.md) | Add Export Hooks | PRD-0303 | PRD-03 |
| 13 | [PRD-0401](PRD-0401.md) | Implement Metrics Core | PRD-0304 | PRD-04 |
| 14 | [PRD-0402](PRD-0402.md) | Implement Custom UAV Evaluator | PRD-0401 | PRD-04 |
| 15 | [PRD-0403](PRD-0403.md) | Implement DUT Evaluator and Report | PRD-0402 | PRD-04 |
| 16 | [PRD-0404](PRD-0404.md) | Add Failure-Case Analysis | PRD-0403 | PRD-04 |
| 17 | [PRD-0501](PRD-0501.md) | Build FastAPI Schemas and App | PRD-0302 | PRD-05 |
| 18 | [PRD-0502](PRD-0502.md) | Wire Prediction Endpoint | PRD-0501 | PRD-05 |
| 19 | [PRD-0503](PRD-0503.md) | Build Docker and Compose Config | PRD-0502 | PRD-05 |
| 20 | [PRD-0504](PRD-0504.md) | Add API and Container Tests | PRD-0503 | PRD-05 |
| 21 | [PRD-0601](PRD-0601.md) | Create ROS2 Message Adapters | PRD-0302 | PRD-06 |
| 22 | [PRD-0602](PRD-0602.md) | Implement ROS2 Detector Node | PRD-0601 | PRD-06 |
| 23 | [PRD-0603](PRD-0603.md) | Add Launch File and Parameters | PRD-0602 | PRD-06 |
| 24 | [PRD-0604](PRD-0604.md) | Add ROS2 Config Tests | PRD-0603 | PRD-06 |
| 25 | [PRD-0701](PRD-0701.md) | Add Runtime Limits and Telemetry | PRD-0504, PRD-0604 | PRD-07 |
| 26 | [PRD-0702](PRD-0702.md) | Add Benchmark Harness | PRD-0701 | PRD-07 |
| 27 | [PRD-0703](PRD-0703.md) | Build Release Manifest | PRD-0702, PRD-0404 | PRD-07 |
| 28 | [PRD-0704](PRD-0704.md) | Add Production Gate Tests | PRD-0703 | PRD-07 |

## Done When

- All 28 tasks are complete.
- `uv run pytest` passes for the module-owned test suite.
- A reproduction report records the custom-UAV and DUT-ANTI-UAV gaps against the paper.
- A releasable checkpoint is available for API and ROS2 use.
