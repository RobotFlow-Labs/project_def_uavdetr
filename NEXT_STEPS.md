# DEF-UAVDETR — Execution Ledger

Resume rule: Read this file COMPLETELY before writing any code.

## 1. Current Status
- **Date**: 2026-04-07
- **Phase**: Round 1 training COMPLETE. Export done. Awaiting 4 GPUs for DDP finetuning.
- **MVP Readiness**: 85%

## 2. What's Done
- [x] PRD-01: Foundation & Config
- [x] PRD-02: Core Model (WTConv + SWSA + ECFRFN + Decoder + Loss)
- [x] PRD-03: Inference Pipeline (CLI + checkpoint I/O + export)
- [x] PRD-05: FastAPI Service + Docker (3-layer serve, CUDA/MLX)
- [x] PRD-06: ROS2 Integration (node + messages + launch)
- [x] PRD-07: Production (telemetry, runtime limits, benchmark, release)
- [x] ANIMA Infrastructure (anima_module.yaml, serve.py, Docker profiles)
- [x] CUDA Kernels (fused wavelet DWT/IDWT, deformable attention)
- [x] Code review: fixed C1 deadlock, H3 target collation, H6 cu128
- [x] Hero page + README
- [x] Round 1 Training: 100 epochs on 75K Seraphim, best val_loss=2.1054
- [x] Export: pth + safetensors + checkpoint → HuggingFace
- [x] Training report + model card
- [x] DDP script ready for 4x L4

## 3. What's Next
- [ ] 4-GPU DDP finetuning on 200K+ superset (Round 2)
- [ ] ONNX export (needs torch update for col2im op)
- [ ] TensorRT FP16/FP32 (generate on target hardware)
- [ ] Round 3 finetune on custom adverse conditions (night/rain/fog)
- [ ] mAP evaluation on test set
- [ ] YOLO26 variant (if DETR results insufficient)

## 4. Datasets Ready
| Dataset | Images | Path | Status |
|---------|--------|------|--------|
| Seraphim | 83K | /mnt/forge-data/datasets/uav_detection/seraphim/ | READY |
| BirdDrone | 86K | /mnt/forge-data/shared_infra/datasets/lat_birddrone/ | READY |
| DroneVehicle-night | 34K | /mnt/forge-data/shared_infra/datasets/dronevehicle_night/ | READY |
| DUT-Anti-UAV | train+val+test | /mnt/train-data/datasets/dut_anti_uav/ | READY |
| **Total** | **200K+** | All YOLO format | **READY** |

## 5. Launch Commands

### Round 2 Finetuning (when 4 GPUs available)
```bash
PYTHONUNBUFFERED=1 PYTHONPATH="" \
torchrun --nproc_per_node=4 --master_port=29500 \
    scripts/train_ddp.py \
    --epochs 30 --batch-size 16 --lr 1e-5 \
    --datasets seraphim,birddrone,dronevehicle_night,dut_anti_uav \
    --resume /mnt/artifacts-datai/checkpoints/project_def_uavdetr/best.pth \
    --workers 2 --patience 10
```

## 6. HuggingFace
- **Repo**: https://huggingface.co/ilessio-aiflowlab/project_def_uavdetr
- **GitHub**: https://github.com/RobotFlow-Labs/project_def_uavdetr
