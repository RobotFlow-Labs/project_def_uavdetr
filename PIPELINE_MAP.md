# DEF-UAVDETR — Pipeline Map

## Scope

This map converts the paper pipeline into the concrete ANIMA build plan for `project_def_uavdetr`. It also records the scaffold mismatch that must be corrected first:

- current scaffold package: `src/anima_inari`
- target package for this module: `src/anima_def_uavdetr`

## Paper Pipeline → Code → PRD

| Paper Component | Paper Section | Reference Repo | Planned File | PRD |
|----------------|---------------|----------------|--------------|-----|
| Frame subsampling, 1-in-5 selection | §3.1, §4.1 | paper only | `src/anima_def_uavdetr/data/sampler.py` | PRD-01 |
| Dataset config and split contract | §4.1 | `datasets/uav_dataset/data.yaml` | `src/anima_def_uavdetr/config.py` | PRD-01 |
| WTConv Block backbone | §3.2 | `ultralytics/nn/extra_modules/wtconv2d.py`, `block.py` | `src/anima_def_uavdetr/models/backbone.py` | PRD-02 |
| SWSA-IFI encoder | §3.3.1 | `ultralytics/nn/extra_modules/SWSATransformer.py` | `src/anima_def_uavdetr/models/encoder.py` | PRD-02 |
| ECFRFN neck (SBA + RepNCSPELAN4) | §3.3.2 | `ultralytics/nn/extra_modules/block.py` | `src/anima_def_uavdetr/models/neck.py` | PRD-02 |
| RT-DETR decoder head | §3.1, §3.5 | `ultralytics/nn/modules/head.py` | `src/anima_def_uavdetr/models/decoder.py` | PRD-02 |
| Inner-CIoU + NWD hybrid loss | §3.4 | paper only | `src/anima_def_uavdetr/losses.py` | PRD-02 |
| End-to-end detector wrapper | §3.1, Algorithm 1 | `ultralytics/cfg/models/UAV-DETR.yaml` | `src/anima_def_uavdetr/model.py` | PRD-02 |
| Offline inference CLI | §4.4 visual results | `detect.py` | `src/anima_def_uavdetr/infer.py`, `scripts/run_infer.py` | PRD-03 |
| Weight loading and export | §4.2, §4.6 | `train.py`, `head.py` | `src/anima_def_uavdetr/checkpoints.py`, `scripts/export.py` | PRD-03 |
| Custom UAV evaluation | §4.3, Table 2 | `val.py` | `src/anima_def_uavdetr/eval/custom_uav.py` | PRD-04 |
| DUT-ANTI-UAV evaluation | §4.4.5, Table 3 | paper only | `src/anima_def_uavdetr/eval/dut_anti_uav.py` | PRD-04 |
| Failure-case visualization | §4.4.3, §4.6 | paper only | `src/anima_def_uavdetr/analysis/failure_cases.py` | PRD-04 |
| FastAPI serving surface | deployment adaptation | n/a | `src/anima_def_uavdetr/api/app.py` | PRD-05 |
| GPU Docker image | deployment adaptation | n/a | `docker/Dockerfile.api`, `docker-compose.api.yml` | PRD-05 |
| ROS2 detector node | ANIMA integration | n/a | `src/anima_def_uavdetr/ros2/node.py` | PRD-06 |
| Launch and topic bridge | ANIMA integration | n/a | `src/anima_def_uavdetr/ros2/launch/uavdetr.launch.py` | PRD-06 |
| Production export, quantization, and release gates | §4.6, §5 | paper + ANIMA requirements | `scripts/release.py`, `scripts/benchmark.py` | PRD-07 |

## Tensor Flow

```text
RGB frame batch
  Tensor[B, 3, 640, 640]
    -> frame sampler (1-of-5 at training time)
    -> WTConv backbone
       S2: Tensor[B,  64, 160, 160]
       S3: Tensor[B, 128,  80,  80]
       S4: Tensor[B, 256,  40,  40]
       S5: Tensor[B, 512,  20,  20]
    -> 1x1 projection + SWSA-IFI encoder
       F5: Tensor[B, 256, 20, 20]
    -> ECFRFN
       P2: Tensor[B, 256, 160, 160]
       P3: Tensor[B, 256,  80,  80]
       P4: Tensor[B, 256,  40,  40]
       P5: Tensor[B, 256,  20,  20]
    -> RT-DETR decoder
       pred_boxes:  Tensor[B, 300, 4]
       pred_logits: Tensor[B, 300, 1]
    -> postprocess
       detections: Tensor[N, 6] = [x1, y1, x2, y2, score, class_id]
```

## Build Order

1. PRD-01 fixes naming, config, and dataset contracts.
2. PRD-02 reproduces the paper architecture and loss.
3. PRD-03 wraps the model in inference and export surfaces.
4. PRD-04 reproduces the tables and visual diagnostics.
5. PRD-05 and PRD-06 add ANIMA deployment surfaces.
6. PRD-07 hardens exports, benchmarks, and release checks.
