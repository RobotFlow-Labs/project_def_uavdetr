# PRD-02: Core Model

> Module: DEF-UAVDETR | Priority: P0
> Depends on: PRD-01
> Status: ✅ Complete

## Objective
Implement the paper-faithful UAV-DETR network path: WTConv backbone, SWSA-IFI encoder, ECFRFN neck, RT-DETR decoder, and Inner-CIoU + NWD training loss.

## Context
UAV-DETR is the heart of the module. The paper’s gains come from a frequency-aware backbone, local-window transformer encoding, selective cross-scale fusion, and a small-object-aware regression objective. This PRD reproduces those pieces without collapsing them into a generic DETR approximation.

Paper references:
- §3.2: WTConv Block for preserving high-frequency structure
- §3.3.1: SWSA-IFI encoder replacing standard AIFI
- §3.3.2: ECFRFN with SBA and RepNCSPELAN4
- §3.4: Inner-CIoU + NWD hybrid loss
- §3.5: Algorithm 1 end-to-end training scheme

## Acceptance Criteria
- [x] The backbone emits `S2..S5` feature maps with the paper-consistent strides.
- [x] The encoder applies SWSA on the projected `S5` feature map and preserves shape `[B, 256, 20, 20]` at `640x640` input.
- [x] The neck emits four fused features `P2..P5`, all `256` channels, suitable for RT-DETR decoding.
- [x] The decoder emits `300` queries and supports training and inference output modes.
- [x] Hybrid regression loss combines Inner-CIoU and NWD with explicit weighting.
- [x] Test: `uv run pytest tests/test_backbone.py tests/test_encoder.py tests/test_neck.py tests/test_model.py tests/test_losses.py -v` passes.
- [x] Benchmark: the implemented architecture is shape-compatible with the reference YAML in `repositories/UAVDETR/ultralytics/cfg/models/UAV-DETR.yaml`.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_def_uavdetr/models/backbone.py` | WTConv block and stage stack | §3.2 | ~220 |
| `src/anima_def_uavdetr/models/encoder.py` | SWSA-IFI encoder | §3.3.1 | ~160 |
| `src/anima_def_uavdetr/models/neck.py` | SBA + RepNCSPELAN4 fusion neck | §3.3.2 | ~220 |
| `src/anima_def_uavdetr/models/decoder.py` | RT-DETR wrapper with 300-query decoder | §3.1, §3.5 | ~180 |
| `src/anima_def_uavdetr/losses.py` | Inner-CIoU + NWD hybrid loss | §3.4 | ~180 |
| `src/anima_def_uavdetr/model.py` | End-to-end detector wrapper | §3.1, Algorithm 1 | ~180 |
| `tests/test_backbone.py` | Backbone shape tests | — | ~120 |
| `tests/test_encoder.py` | Encoder shape tests | — | ~80 |
| `tests/test_neck.py` | Neck shape tests | — | ~100 |
| `tests/test_model.py` | End-to-end forward tests | — | ~120 |
| `tests/test_losses.py` | Loss behavior tests | — | ~100 |

## Architecture Detail

### Inputs
- `images: Tensor[B, 3, 640, 640]`
- `targets["boxes"]: Tensor[M, 4]`
- `targets["labels"]: Tensor[M, 1]`

### Outputs
- `S2: Tensor[B, 64, 160, 160]`
- `S3: Tensor[B, 128, 80, 80]`
- `S4: Tensor[B, 256, 40, 40]`
- `S5: Tensor[B, 512, 20, 20]`
- `P2..P5: Tensor[B, 256, H_i, W_i]`
- `pred_boxes: Tensor[B, 300, 4]`
- `pred_logits: Tensor[B, 300, 1]`
- `loss_dict: dict[str, Tensor]`

### Algorithm
```python
# Paper §3.5 — Algorithm 1
class DefUavDetr(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = WTConvBackbone(cfg.backbone)
        self.encoder = SWSAIFIEncoder(dim=256, heads=8, patch_size=4, stride=4)
        self.neck = ECFRFN(cfg.neck)
        self.decoder = RTDETRHead(num_classes=1, num_queries=300, num_decoder_layers=3)
        self.loss_fn = HybridInnerCiouNwdLoss(alpha=cfg.loss.alpha, nwd_constant=cfg.loss.nwd_constant)

    def forward(self, images, targets=None):
        s2, s3, s4, s5 = self.backbone(images)
        f5 = self.encoder(project_to_256(s5))
        p2, p3, p4, p5 = self.neck(s2, s3, s4, f5)
        pred_boxes, pred_logits = self.decoder((p2, p3, p4, p5), targets=targets)
        if targets is None:
            return pred_boxes, pred_logits
        return self.loss_fn(pred_boxes, pred_logits, targets)
```

## Dependencies
```toml
torch = ">=2.0"
einops = ">=0.8"
pywavelets = ">=1.6"
```

## Data Requirements
| Asset | Size | Path | Download |
|------|------|------|----------|
| Vendored reference modules | repo local | `repositories/UAVDETR/ultralytics/nn/` | READY |
| Custom UAV labels | dataset-dependent | `/Volumes/AIFlowDev/RobotFlowLabs/datasets/def_uavdetr/uav_dataset/` | Required for training |

## Test Plan
```bash
uv run pytest tests/test_backbone.py tests/test_encoder.py tests/test_neck.py tests/test_model.py tests/test_losses.py -v
```

## References
- Paper: §3.2 "WTConv Block"
- Paper: §3.3.1 "SWSA-IFI Encoder"
- Paper: §3.3.2 "ECFRFN Module"
- Paper: §3.4 "InnerCIoU-NWD Hybrid Loss"
- Paper: §3.5 "Pseudo Code"
- Reference impl: `repositories/UAVDETR/ultralytics/cfg/models/UAV-DETR.yaml`
- Reference impl: `repositories/UAVDETR/ultralytics/nn/extra_modules/block.py`
- Reference impl: `repositories/UAVDETR/ultralytics/nn/extra_modules/SWSATransformer.py`
- Feeds into: PRD-03, PRD-04
