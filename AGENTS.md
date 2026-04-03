# DEF-UAVDETR

## Paper
**UAV-DETR: Anti-Drone Detection**
arXiv: https://arxiv.org/abs/2603.22841

## Module Identity
- Codename: DEF-UAVDETR
- Domain: Defense
- Part of ANIMA Intelligence Compiler Suite

## Structure
```
project_def_uavdetr/
├── pyproject.toml
├── configs/
├── src/anima_def_uavdetr/
├── tests/
├── scripts/
├── papers/          # Paper PDF
├── AGENTS.md        # This file
├── NEXT_STEPS.md
├── ASSETS.md
└── PRD.md
```

## Commands
```bash
uv sync
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Conventions
- Package manager: uv (never pip)
- Build backend: hatchling
- Python: >=3.10
- Config: TOML + Pydantic BaseSettings
- Lint: ruff
- Git commit prefix: [DEF-UAVDETR]
