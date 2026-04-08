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

### v1 (Round 1)
| Dataset | Images | Source | License |
|---------|--------|--------|---------|
| Seraphim | 75,134 (train) | 23 open sources | CC BY 4.0 |

### v2 (Round 2 — current best)
| Dataset | Images | Source |
|---------|--------|--------|
| Seraphim | 75,134 | Curated drone detection |
| BirdDrone | 145,506 | Drone + bird video frames |
| DUT-Anti-UAV | 5,200 | 35 UAV types benchmark |
| DroneVehicle-night | 10,357 | Night IR aerial imagery |
| **Total train** | **224,388** | |
| **Total val** | **11,809** | |

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

| Version | Dataset | Images | Epochs | Hardware | Val Loss | Status |
|---------|---------|--------|--------|----------|----------|--------|
| v1 | Seraphim | 75K | 100 | 1x L4 | 2.1054 | Released |
| v2 | Unified (6 datasets) | 236K | 30 | 4x L4 DDP | 2.3017 | Released |
| v3 | Unified v2 (7 datasets) | 261K | 20 | 4x L4 DDP | — | Ready |
| v4 | Nighthawk Mega | 1.78M | 20 | 8x L4 DDP | — | Planned |

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
