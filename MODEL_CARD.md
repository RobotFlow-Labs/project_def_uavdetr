# Model Card: DEF-UAVDETR

## Model Details

- **Model name**: DEF-UAVDETR v1
- **Model type**: Object detection (single-class UAV/drone)
- **Architecture**: RT-DETR + WTConv backbone + SWSA-IFI encoder + ECFRFN neck
- **Parameters**: 11,496,174 (11.5M)
- **Input**: RGB 640x640
- **Output**: Bounding boxes [x1, y1, x2, y2, score, class_id], up to 300 detections
- **Paper**: UAV-DETR: Anti-Drone Detection ([arXiv:2603.22841](https://arxiv.org/abs/2603.22841))
- **Authors**: Jun Yang, Dong Wang, Hongxu Yin, Hongpeng Li, Jianxiong Yu
- **License**: Apache 2.0
- **Organization**: Robot Flow Labs / AIFLOW LABS LIMITED

## Intended Use

- **Primary use**: Real-time UAV/drone detection in RGB video streams
- **Target deployment**: Counter-UAS systems, airspace security, perimeter defense
- **Platforms**: NVIDIA GPUs (L4, Jetson, A100), CPU fallback supported
- **Integration**: FastAPI service, ROS2 node, Docker container

## Training Data

| Dataset | Images | Source | License |
|---------|--------|--------|---------|
| Seraphim | 75,134 (train) | 23 open sources | CC BY 4.0 |
| Seraphim | 3,756 (val) | — | CC BY 4.0 |
| Seraphim | 8,349 (test) | — | CC BY 4.0 |

**Not yet trained on**: DUT-Anti-UAV, BirdDrone, DroneVehicle-night (planned for Round 2)

## Training Procedure

- **Epochs**: 100
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: Cosine with 5% linear warmup
- **Precision**: FP16 (automatic mixed precision)
- **Batch size**: 16
- **Hardware**: 1x NVIDIA L4 (23GB)
- **Duration**: ~100 hours

## Performance

| Metric | Value |
|--------|-------|
| Best val_loss | 2.1054 |
| Final train_loss | 1.68 |
| Train/val gap | 0.43 (no severe overfitting) |

**Note**: Detection metrics (mAP, precision, recall) require running the evaluation pipeline on the test set. The paper targets mAP50:95 of 62.56% on the custom UAV dataset.

## Limitations

Documented in the paper (Section 4.6):

- **Bird-like distractors**: May produce false positives on birds, kites, balloons
- **Urban camouflage**: Reduced recall when drones are against cluttered urban backgrounds
- **FLOPs overhead**: 17.2% more compute than baseline RT-DETR
- **Adverse conditions**: Not specifically trained for night, rain, or fog (planned for Round 3)
- **Single class**: Only detects "UAV" — no drone type classification

## Ethical Considerations

- This model is designed for **defensive** counter-UAS applications (airspace security, critical infrastructure protection)
- Detection capabilities should be deployed within applicable legal frameworks for drone surveillance
- False positives on birds/kites should be considered when setting confidence thresholds for automated response systems

## Model Versions

| Version | Dataset | Epochs | Val Loss | Status |
|---------|---------|--------|----------|--------|
| v1 (this) | Seraphim 75K | 100 | 2.1054 | Released |
| v2 (planned) | 200K+ superset | 30 | — | Pending 4-GPU DDP |
| v3 (planned) | + night/rain/fog | 20 | — | Pending custom data |

## Citation

```bibtex
@article{yang2025uavdetr,
  title={UAV-DETR: Anti-Drone Detection},
  author={Yang, Jun and Wang, Dong and Yin, Hongxu and Li, Hongpeng and Yu, Jianxiong},
  journal={arXiv preprint arXiv:2603.22841},
  year={2025}
}
```

## Contact

- **Organization**: [Robot Flow Labs](https://robotflow-labs.github.io)
- **HuggingFace**: [ilessio-aiflowlab/project_def_uavdetr](https://huggingface.co/ilessio-aiflowlab/project_def_uavdetr)
- **GitHub**: [RobotFlow-Labs/project_def_uavdetr](https://github.com/RobotFlow-Labs/project_def_uavdetr)
