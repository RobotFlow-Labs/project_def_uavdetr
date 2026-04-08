# DEF-UAVDETR — Training Report

## Training Rounds

| Round | Dataset | Images | Epochs | Hardware | Best val_loss | Time |
|-------|---------|--------|--------|----------|--------------|------|
| **R1** | Seraphim | 75K | 100 | 1x L4 | 2.1054 | ~100h |
| **R2** | Unified (6 datasets) | 236K | 30 | 4x L4 DDP | 2.3017 | ~5h |
| **R3** | Unified v2 (7 datasets) | 261K | — | Planned | — | — |
| **R4** | Nighthawk Mega | 1.78M | — | 8x L4 DDP | — | — |

---

## Round 1 — Baseline (Seraphim)

| Metric | Value |
|--------|-------|
| **Model** | DefUavDetr (RT-DETR + WTConv + SWSA-IFI + ECFRFN) |
| **Parameters** | 11,496,174 (11.5M) |
| **Dataset** | Seraphim (75,134 train / 3,756 val) |
| **Epochs** | 100 |
| **Best val_loss** | 2.1054 (epoch 99) |
| **Final train_loss** | 1.68 |
| **Training time** | ~100 hours (with 3 crash-restarts) |
| **Hardware** | 1x NVIDIA L4 (23GB VRAM) |
| **Batch size** | 16 |
| **Speed** | 1.5 steps/s |
| **Checkpoint** | def_uavdetr_v1.pth (46MB) |

### R1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 (peak) |
| Weight decay | 1e-4 |
| Scheduler | Cosine with 5% warmup |
| Min LR | 1e-7 |
| Precision | FP16 (AMP) |
| Gradient clipping | max_norm=1.0 |
| Seed | 42 |

### R1 Loss Curve

```
Epoch  Train   Val     Phase
  1    5.12    4.92    Warmup (LR ramp 0 → 1e-4)
  5    4.42    4.21    Warmup complete, peak LR
 10    3.33    3.28    Rapid descent
 20    2.99    2.96    
 30    2.77    2.80    
 40    2.62    2.66    
 50    2.39    2.56    
 60    2.23    2.43    
 70    1.95    2.25    
 80    1.82    2.21    
 90    1.72    2.17    
 99    1.68    2.11    ← Best val_loss
100    1.68    2.15    Final epoch
```

### R1 Crash Log

| Event | Epoch | Cause | Recovery |
|-------|-------|-------|----------|
| Crash 1 | 48 | pin_memory thread exit | Fixed pin_memory=False, resumed |
| Crash 2 | 86 | Silent OOM kill | Resumed from best.pth |
| Crash 3 | 92 | Silent OOM kill | Resumed from best.pth |

---

## Round 2 — DDP Finetuning (236K Unified)

| Metric | Value |
|--------|-------|
| **Dataset** | Unified: Seraphim + BirdDrone + DroneVehicle-night + DUT-Anti-UAV |
| **Train images** | 224,388 |
| **Val images** | 11,809 |
| **Epochs** | 30 (crashed at epoch 29, val plateaued) |
| **Best val_loss** | 2.3017 (epoch 27) |
| **Final train_loss** | 2.28 |
| **Training time** | ~5 hours |
| **Hardware** | 4x NVIDIA L4 DDP (92GB total VRAM) |
| **Effective batch** | 64 (16 per GPU x 4) |
| **Speed** | 5.9 eff steps/s |
| **Checkpoint** | def_uavdetr_v2.pth (46MB) |

### R2 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Base learning rate | 1e-5 |
| Scaled LR (4x DDP) | 4e-5 |
| Weight decay | 1e-4 |
| Scheduler | Cosine with 5% warmup |
| Min LR | 1e-7 |
| Precision | FP16 (AMP) |
| Gradient clipping | max_norm=1.0 |
| Resume from | R1 best.pth (val_loss=2.1054) |

### R2 Loss Curve

```
Epoch  Train   Val     Phase
  1    4.07    3.58    Adapting to new data distribution
  5    3.01    2.93    Warmup done, rapid adaptation
 10    2.78    2.81    
 15    2.56    2.58    
 20    2.40    2.40    
 25    2.29    2.33    
 27    2.28    2.30    ← Best val_loss
 28    2.28    2.31    Plateauing
```

### R2 Dataset Breakdown

| Dataset | Images | Source | Format |
|---------|--------|--------|--------|
| Seraphim | 75,134 | 23 open sources | YOLO native |
| BirdDrone | 145,506 | Drone + bird frames | YOLO native |
| DUT-Anti-UAV | 5,200 | Public benchmark | Converted from VOC XML |
| DroneVehicle-night | 10,357 | Night IR imagery | Converted from DOTA |
| **Total train** | **224,388** | | |
| **Total val** | **11,809** | | |

### R2 Key Observations

1. **Initial loss spike**: Val went from 2.10 → 3.58 at epoch 1 (expected — new data distribution)
2. **Fast adaptation**: Dropped from 3.58 → 2.30 in 27 epochs
3. **Val loss higher than R1**: 2.30 vs 2.10 — because the dataset is 3x harder (birds, night, vehicles)
4. **No overfitting**: Train/val gap stayed at ~0.03 (very healthy for 236K images)
5. **Crash at epoch 29**: Silent OOM, but val was plateaued — no loss of quality

---

## Comparison: R1 vs R2

| Metric | R1 (Seraphim) | R2 (Unified DDP) |
|--------|--------------|------------------|
| Dataset | 75K (drones only) | 236K (drones + birds + vehicles + night) |
| Best val_loss | 2.1054 | 2.3017 |
| Train/val gap | 0.43 | 0.03 |
| Generalization | Single domain | Multi-domain |
| Training time | 100 hours | 5 hours |
| Hardware | 1x L4 | 4x L4 DDP |
| Checkpoint | v1 | v2 |

**Note**: R2 val_loss is higher because the validation set includes harder examples (birds, night conditions, vehicles). The model generalizes much better despite the higher loss number.

---

## Export Artifacts

| Format | v1 (R1) | v2 (R2) | Status |
|--------|---------|---------|--------|
| PyTorch .pth | 46 MB | 46 MB | Ready |
| SafeTensors | 46 MB | 46 MB | Ready |
| Checkpoint | 133 MB | 133 MB | Ready |
| ONNX | — | — | Deferred (col2im op) |
| TensorRT FP16 | — | — | Deferred (target HW) |
| TensorRT FP32 | — | — | Deferred (target HW) |

## Planned Rounds

### Round 3 — Unified v2 (261K)
- Add: Baidu UAV (10.3K) + VisDrone (8.6K)
- 20 epochs, lr=5e-6, 4x L4 DDP
- Resume from R2 best.pth

### Round 4 — Nighthawk Mega (1.78M)
- 5 conditions: night, thermal, fog, rain, dusk + original
- 20 epochs, lr=5e-6, 8x L4 DDP
- YOLO + COCO + SAM2 masks + PaliGemma captions
- Shared foundation for all 11 defense modules

## HuggingFace

- **Repo**: [ilessio-aiflowlab/project_def_uavdetr](https://huggingface.co/ilessio-aiflowlab/project_def_uavdetr)
- **Last pushed**: 2026-04-08
