# Storage Layout (AI_WAREHOUSE 3.0)

This repository is designed to run on an AI workstation that follows the **AI_WAREHOUSE 3.0** filesystem layout.

## Defaults

- **Code and models:** `/mnt/c`
  - Models: `/mnt/c/ai_models`
  - Projects: `/mnt/c/ai_projects`
  - Caches: `/mnt/c/ai_cache`
- **Datasets and training outputs:** `/mnt/data`
  - Datasets: `/mnt/data/datasets`
  - Training runs: `/mnt/data/training/runs`
  - Training logs: `/mnt/data/training/logs`

## Cache Environment Variables

Dev scripts and notebooks should set these to avoid writing large cache files under `$HOME/.cache`:

```bash
export HF_HOME=/mnt/c/ai_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/c/ai_cache/huggingface
export TORCH_HOME=/mnt/c/ai_cache/torch
export XDG_CACHE_HOME=/mnt/c/ai_cache
export PIP_CACHE_DIR=/mnt/c/ai_cache/pip
```

## EdgeDeID Defaults

By default, this project resolves model/dataset locations using the storage layout:

- Models: `/mnt/c/ai_models/detection/edge_deid/...`
- Prepared datasets: `/mnt/data/datasets/edge_deid/processed/...`
- Training runs: `/mnt/data/training/runs/edge_deid/...`

Override points (optional):
- `EDGE_DEID_CACHE_HOME`
- `EDGE_DEID_MODELS_HOME`
- `EDGE_DEID_DATA_HOME`
- `EDGE_DEID_DATASETS_HOME`
- `EDGE_DEID_TRAINING_HOME`
