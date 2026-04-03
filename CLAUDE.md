# INARI — Agent Instructions

## Identity
You are the INARI agent. You own exactly ONE paper.
You must verify this paper is real and reproducible before building anything.
You are part of Wave-7 with 16 sibling projects running in parallel.
Wave-7 theme: Japanese Mythology.

## Your Paper
- **Title**: UAV-DETR: Anti-Drone Detection
- **ArXiv**: 2603.22841
- **Link**: https://arxiv.org/abs/2603.22841
- **Repo**: https://github.com/wd-sir/UAVDETR
- **Compute**: GPU-NEED
- **Functional Name**: DEF-uavdetr
- **Stack Fit**: Defense
- **Core method**: (TODO — fill after reading paper)
- **What we take**: (TODO)
- **What we skip**: (TODO)
- **What we adapt**: (TODO)

## Verification Checklist (DO THIS FIRST)
1. Read the paper completely
2. Check if reference repo exists and runs
3. Check if reported datasets are accessible
4. Check if claimed metrics are plausible
5. Look for independent reproductions or citations
6. If ANY red flag → document in NEXT_STEPS.md and flag for CTO review
7. If paper is unusable → create KILLED.md with reason, stop work

## Your Datasets
- Check shared volume first: /Volumes/AIFlowDev/RobotFlowLabs/datasets
- Download script: `bash scripts/download_data.sh`
- Repo clone location: /Volumes/AIFlowDev/RobotFlowLabs/repos/wave7

## Hardware
- **ZED 2i stereo camera**: Available — SDK at /usr/local/zed/
- **Unitree L2 3D LiDAR**: Available
- **xArm 6 cobot**: Pending purchase (~$5K)
- **Mac Studio M-series**: MLX development
- **8x RTX 6000 Pro Blackwell**: GCloud GPU training

## Infrastructure
- **Datasets**: /Volumes/AIFlowDev/RobotFlowLabs/datasets
- **ROS2 Bridge**: /Users/ilessio/Development/AIFLOWLABS/projects/ROS2
- **GPU Server**: GCloud (8x RTX 6000 Pro Blackwell)

## Dual-Compute Mandate
ALL code MUST run on BOTH:
- `ANIMA_BACKEND=mlx` (Apple Silicon, no CUDA)
- `ANIMA_BACKEND=cuda` (GPU server, no MLX)
Use `src/anima_inari/device.py` for backend abstraction.

## Multi-Agent Protocol
You are 1 of 17 agents working in parallel on Wave-7.
1. NEVER touch files outside `project_inari/`
2. Prefix every commit with `[INARI]`
3. Stage only your own files
4. Shared datasets — READ ONLY, never modify
5. If you need output from another project, document the dependency — do NOT import
6. Read NEXT_STEPS.md before every session

## All Sibling Projects
| # | Folder | Codename | Functional | Paper | Compute | Stack |
|---|--------|----------|------------|-------|---------|-------|
| 1 | project_amaterasu | AMATERASU | SLAM-coko | CokO-SLAM: Multi-Agent Collaborative GS SLAM | GPU-NEED | ATLAS |
| 2 | project_tsukuyomi | TSUKUYOMI | SLAM-gs3lam | GS3LAM: Gaussian Semantic SLAM | GPU-NEED | PHANTOMAGIA |
| 3 | project_susanoo | SUSANOO | SLAM-mipslam | MipSLAM: Anti-Aliased 3DGS SLAM | GPU-NEED | PHANTOMAGIA |
| 4 | project_raijin | RAIJIN | MANIP-ultradex | UltraDexGrasp: Bimanual Dexterous Grasping | GPU-NEED | PROMETHEUS |
| 5 | project_fujin | FUJIN | GS-ghost | GHOST: Hand-Object Reconstruction via 3DGS | GPU-NEED | PROMETHEUS |
| 6 | project_hachiman | HACHIMAN | DEF-rpga | Robust Physical-World Adversarial Camouflage via 3DGS | GPU-NEED | Defense |
| 7 | project_bishamonten | BISHAMONTEN | DEF-dtp | DTP-Attack: Trajectory Prediction Attack | MLX-OK | ATLAS |
| 8 | project_daikokuten | DAIKOKUTEN | DEF-cai | CAI: Robot Cybersecurity Framework | MLX-OK | NIDHOGG |
| 9 | project_tengu | TENGU | UAV-trackvla | UAV-Track VLA: Embodied Aerial Tracking | GPU-NEED | Defense |
| 10 | project_izanagi | IZANAGI | VIS-occany | OccAny: Generalized Urban 3D Occupancy (CVPR 2026) | GPU-NEED | PHANTOMAGIA |
| 11 | project_kodama | KODAMA | VIS-forestsim | ForestSim: Off-Road Segmentation Benchmark | GPU-NEED | NEMESIS |
| 12 | project_shinigami | SHINIGAMI | DEF-ghostfwl | Ghost-FWL: LiDAR Ghost Object Detection | MLX-OK | Defense |
| 13 | project_ebisu | EBISU | CALIB-projfusion | ProjFusion: Camera-LiDAR Calibration (RA-L 2026) | GPU-NEED | ALL |
| 14 | project_ryujin | RYUJIN | SIM-carlaair | CARLA-Air: Air-Ground Unified Sim | GPU-NEED | Defense-ATLAS |
| 15 | project_benzaiten | BENZAITEN | MANIP-raap | RAAP: Affordance Prediction (ICRA 2026) | GPU-NEED | PROMETHEUS |
| 16 | project_inari | INARI | DEF-uavdetr | UAV-DETR: Anti-Drone Detection | GPU-NEED | Defense |
| 17 | project_kaguya | KAGUYA | GS-ovie | OVIE: Open-Vocabulary Novel View Synthesis | GPU-NEED | PHANTOMAGIA |
