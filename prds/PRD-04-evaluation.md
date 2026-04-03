# PRD-04: Evaluation & Reproduction

> Module: DEF-UAVDETR | Priority: P1
> Depends on: PRD-02, PRD-03
> Status: ⬜ Not started

## Objective
Reproduce the paper’s evaluation protocol, generate the key custom-UAV and DUT-ANTI-UAV metrics, and emit comparison artifacts that clearly show any gap versus the paper.

## Context
The paper’s value is not just the architecture but the measured uplift: `62.56 mAP50:95` on the custom UAV dataset and `67.15 mAP50:95` on DUT-ANTI-UAV. This PRD creates the evaluation harness, report generation, and failure-case tooling needed to validate those claims.

Paper references:
- §4.1: custom dataset curation and `7:2:1` split
- §4.2: paper comparison protocol uses `100` epochs, scratch training for the primary models
- §4.3: evaluation metrics
- §4.4.2: Table 2 custom-UAV results
- §4.4.5: Table 3 DUT-ANTI-UAV results
- §4.6: false positives from birds and false negatives under camouflage

## Acceptance Criteria
- [ ] Custom UAV evaluation computes Precision, Recall, F1, mAP50, mAP75, and mAP50:95.
- [ ] DUT-ANTI-UAV evaluation computes the same metrics and stores per-run summaries.
- [ ] A paper-comparison report records the measured gap against Table 2 and Table 3.
- [ ] Failure-case scripts export bird-confusion and camouflage examples for manual review.
- [ ] Test: `uv run pytest tests/test_metrics.py tests/test_eval_reports.py -v` passes.
- [ ] Benchmark: a generated markdown report includes the paper targets `62.56` and `67.15` for `mAP50:95`.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_def_uavdetr/metrics.py` | Metric computation helpers | §4.3 | ~140 |
| `src/anima_def_uavdetr/eval/custom_uav.py` | Custom UAV evaluator | §4.1, Table 2 | ~180 |
| `src/anima_def_uavdetr/eval/dut_anti_uav.py` | DUT evaluator | §4.4.5, Table 3 | ~180 |
| `src/anima_def_uavdetr/analysis/failure_cases.py` | FP/FN artifact extraction | §4.6 | ~140 |
| `scripts/run_eval.py` | Unified evaluation CLI | §4.3 | ~100 |
| `scripts/build_report.py` | Markdown comparison report | Table 2, Table 3 | ~120 |
| `tests/test_metrics.py` | Metric tests | — | ~120 |
| `tests/test_eval_reports.py` | Report tests | — | ~100 |

## Architecture Detail

### Inputs
- `predictions: list[Tensor[N_i, 6]]`
- `ground_truth: list[Tensor[M_i, 5]]`
- `dataset_name: Literal["custom_uav", "dut_anti_uav"]`

### Outputs
- `metrics: dict[str, float]`
- `comparison_report: Path`
- `failure_case_manifest: Path`

### Algorithm
```python
PAPER_TARGETS = {
    "custom_uav": {"map50_95": 62.56, "precision": 96.82, "f1": 95.87},
    "dut_anti_uav": {"map50_95": 67.15, "precision": 97.09, "f1": 95.26},
}


def summarize_run(dataset_name: str, metrics: dict[str, float]) -> dict[str, float]:
    targets = PAPER_TARGETS[dataset_name]
    return {f"gap_{k}": metrics[k] - targets[k] for k in targets}
```

## Dependencies
```toml
pandas = ">=2.2"
matplotlib = ">=3.9"
```

## Data Requirements
| Asset | Size | Path | Download |
|------|------|------|----------|
| Custom UAV test split | 10% of custom set | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/def_uavdetr/uav_dataset/test/` | Required |
| DUT-ANTI-UAV eval split | benchmark-dependent | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/def_uavdetr/dut_anti_uav/` | Required |

## Test Plan
```bash
uv run pytest tests/test_metrics.py tests/test_eval_reports.py -v
```

## References
- Paper: §4.1 "Dataset Preparation"
- Paper: §4.3 "Evaluation Metrics"
- Paper: Table 2 "custom UAV dataset"
- Paper: Table 3 "DUT-ANTI-UAV benchmark"
- Paper: §4.6 "Algorithm Failures and Limitations"
- Reference impl: `repositories/UAVDETR/val.py`
- Feeds into: PRD-07
