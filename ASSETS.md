# DEF-UAVDETR — Asset Manifest

## Paper
- Title: UAV-DETR: DETR for Anti-Drone Target Detection
- ArXiv: 2603.22841
- PDF: `papers/2603.22841_UAV-DETR.pdf`
- Authors: Jun Yang, Dong Wang, Hongxu Yin, Hongpeng Li, Jianxiong Yu
- Reference repo: `repositories/UAVDETR`

## Status: ALMOST

The paper PDF and a vendored reference repo are present locally. The datasets and paper-faithful checkpoints are not staged on the shared volume yet. The current scaffold still uses the stale `INARI` codename and package paths; PRD-01 fixes that before model work starts.

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|-------|------|--------|---------------|--------|
| UAV-DETR scratch init | n/a | Paper §4.2 says the primary protocol trains from scratch | n/a | READY |
| `uavdetr-best.pt` | unknown | Not bundled in the repo; expected to be produced by reproduction runs | `/Volumes/AIFlowDev/RobotFlowLabs/models/def_uavdetr/checkpoints/uavdetr-best.pt` | MISSING |
| RT-DETR baseline checkpoint (optional) | unknown | Optional baseline-only convenience weight, not required for paper-faithful training | `/Volumes/AIFlowDev/RobotFlowLabs/models/def_uavdetr/checkpoints/rtdetr-baseline.pt` | MISSING |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---------|------|-------|--------|------|--------|
| Custom UAV dataset | 14,713 RGB images | 7:2:1 train/val/test | Paper §4.1, plus repo `datasets/uav_dataset/data.yaml` layout | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/def_uavdetr/uav_dataset/` | MISSING |
| DUT-ANTI-UAV | paper size not stated | official test protocol | Public benchmark used in paper §4.4.5 | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/def_uavdetr/dut_anti_uav/` | MISSING |

## Reference Code Assets
| Asset | Purpose | Local Path | Status |
|------|---------|-----------|--------|
| Vendored reference repo | Ground-truth implementation for architecture and training defaults | `repositories/UAVDETR` | READY |
| Model YAML | Paper-faithful module graph | `repositories/UAVDETR/ultralytics/cfg/models/UAV-DETR.yaml` | READY |
| Training entrypoint | Confirms repo runtime defaults | `repositories/UAVDETR/train.py` | READY |
| Evaluation entrypoint | Confirms repo test split and batch defaults | `repositories/UAVDETR/val.py` | READY |

## Hyperparameters
| Param | Value | Source |
|-------|-------|--------|
| frame_sampling | 1 image randomly sampled from every 5 adjacent frames | Paper §3.1, §4.1 |
| input_resolution | `640x640` | Repo `train.py`, `val.py`, `ultralytics/cfg/default.yaml` |
| epochs | `100` for paper comparisons | Paper §4.2 |
| repo_epochs_default | `300` | Repo `train.py` |
| optimizer | `AdamW` | Repo `ultralytics/cfg/default.yaml` |
| learning_rate | `1e-4` | Repo `ultralytics/cfg/default.yaml` |
| batch_size | `16` in repo launcher, `4` in default config | Repo `train.py`, `ultralytics/cfg/default.yaml` |
| weight_decay | `1e-4` | Repo `ultralytics/cfg/default.yaml` |
| warmup | `2000` warmup iterations | Repo `ultralytics/cfg/default.yaml` |
| workers | `4` | Repo `train.py`, `ultralytics/cfg/default.yaml` |
| max_detections | `300` queries/detections | Repo `UAV-DETR.yaml`, `default.yaml` |

## Architecture Facts
| Component | Paper Section | Reference |
|----------|---------------|-----------|
| WTConv-enhanced backbone | §3.2 | `ultralytics/nn/extra_modules/wtconv2d.py`, `block.py` |
| SWSA-IFI encoder | §3.3.1 | `ultralytics/nn/extra_modules/SWSATransformer.py` |
| ECFRFN neck with SBA + RepNCSPELAN4 | §3.3.2 | `ultralytics/nn/extra_modules/block.py` |
| RT-DETR decoder, 300 queries | §3.1, §3.5 | `ultralytics/nn/modules/head.py` |
| Inner-CIoU + NWD hybrid loss | §3.4 | Paper only; repo integrates through the modified RT-DETR training stack |

## Expected Metrics
| Benchmark | Metric | Paper Value | Our Target |
|-----------|--------|-------------|-----------|
| Custom UAV dataset | Precision | `96.82%` | `>= 96.0%` |
| Custom UAV dataset | Recall | `94.93%` | `>= 94.0%` |
| Custom UAV dataset | F1 | `95.87%` | `>= 95.0%` |
| Custom UAV dataset | mAP50 | `96.58%` | `>= 95.5%` |
| Custom UAV dataset | mAP75 | `71.08%` | `>= 69.0%` |
| Custom UAV dataset | mAP50:95 | `62.56%` | `>= 60.0%` |
| Custom UAV dataset | Params | `11,962,040` | `<= 12.1M` |
| DUT-ANTI-UAV | Precision | `97.09%` | `>= 96.5%` |
| DUT-ANTI-UAV | F1 | `95.26%` | `>= 94.5%` |
| DUT-ANTI-UAV | mAP50:95 | `67.15%` | `>= 65.5%` |

## Environment
| Component | Value | Source |
|----------|-------|--------|
| OS | Ubuntu 20.04.6 LTS | Paper Table 1 |
| CPU | Intel i7-12700KF | Paper Table 1 |
| RAM | 64 GB | Paper Table 1 |
| GPU | NVIDIA RTX 3090 24 GB | Paper Table 1 |
| Python | 3.9.25 | Paper Table 1 |
| PyTorch | 1.12.1 | Paper Table 1 |
| CUDA | 11.3 | Paper Table 1 |
| cuDNN | 8.3.2 | Paper Table 1 |

## Reproduction Notes
- The paper’s main comparison protocol trains all core models from scratch for `100` epochs.
- The vendored repo launcher uses `300` epochs. PRD-04 keeps both paths explicit and treats the `100` epoch protocol as the paper-faithful baseline.
- The current project scaffold points at `src/anima_inari`; PRD-01 retargets the module to `src/anima_def_uavdetr`.
