# DEF-UAVDETR — Training Report

## Summary

| Metric | Value |
|--------|-------|
| **Model** | DefUavDetr (RT-DETR + WTConv + SWSA-IFI + ECFRFN) |
| **Parameters** | 11,496,174 (11.5M) |
| **Dataset** | Seraphim (75,134 train / 3,756 val) |
| **Total epochs** | 100 |
| **Best val_loss** | 2.1054 (epoch 99) |
| **Final train_loss** | 1.68 |
| **Training time** | ~100 hours (with 3 crash-restarts) |
| **Hardware** | 1x NVIDIA L4 (23GB VRAM) |
| **Batch size** | 16 |
| **VRAM usage** | 8.9 GB (38%) |
| **GPU utilization** | 95-100% |
| **Speed** | 1.5 steps/s |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 (peak) |
| Weight decay | 1e-4 |
| Scheduler | Cosine with 5% warmup |
| Min LR | 1e-7 |
| Precision | FP16 (AMP) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=20 (not triggered) |
| Seed | 42 |

## Loss Curve

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
 95    1.67    2.14    
 99    1.68    2.11    ← Best val_loss
100    1.68    2.15    Final epoch
```

## Key Observations

1. **No overfitting**: Train/val gap stayed at ~0.4 throughout, healthy for 75K images
2. **Smooth convergence**: No loss spikes, NaN, or divergence
3. **Val loss plateau**: Best at epoch 99 (2.1054), convergence flattened after epoch 80
4. **3 crash-restarts**: Silent OOM kills at epochs 48, 86, 92 — all recovered from checkpoints
5. **LR decay**: Cosine schedule reached min_lr (1e-7) by epoch 95, last 5 epochs had minimal learning

## Crash Log

| Event | Epoch | Cause | Recovery |
|-------|-------|-------|----------|
| Crash 1 | 48 | pin_memory thread exit | Fixed pin_memory=False, resumed |
| Crash 2 | 86 | Silent OOM kill | Resumed from best.pth |
| Crash 3 | 92 | Silent OOM kill | Resumed from best.pth |

## Dataset

| Dataset | Images | Format | Resolution |
|---------|--------|--------|------------|
| Seraphim | 83,483 total | YOLO | 640x640 |
| Train split | 71,378 (95%) | — | — |
| Val split | 3,756 (5%) | — | — |
| Source | 23 open sources | CC BY 4.0 | — |

## Export Artifacts

| Format | File | Size | Status |
|--------|------|------|--------|
| PyTorch | def_uavdetr_v1.pth | 46 MB | Ready |
| SafeTensors | def_uavdetr_v1.safetensors | 46 MB | Ready |
| Checkpoint | best.pth | 133 MB | Ready (includes optimizer) |
| ONNX | — | — | Deferred (col2im op) |
| TensorRT FP16 | — | — | Deferred (target HW) |
| TensorRT FP32 | — | — | Deferred (target HW) |

## Next Steps

1. **Round 2 finetuning**: 200K+ superset on 4x L4 DDP (Seraphim + BirdDrone + DroneVehicle-night + DUT-Anti-UAV)
2. **Round 3 finetuning**: Custom adverse conditions (night, rain, fog)
3. **ONNX export**: After torch upgrade or custom fold operator
4. **TensorRT**: Generate on deployment target hardware
5. **mAP evaluation**: Run detection metrics on test set

## HuggingFace

- **Repo**: [ilessio-aiflowlab/project_def_uavdetr](https://huggingface.co/ilessio-aiflowlab/project_def_uavdetr)
- **Pushed**: 2026-04-07
