# Local Training & Fine-Tuning (Dev-only)

This repository is **offline-first at runtime**. Dataset ingestion and model fine-tuning are **dev-only** workflows and require explicit opt-in for network access.

## 1) Environment and Storage Layout

The default storage layout follows AI_WAREHOUSE 3.0:
- Cache: `/mnt/c/ai_cache`
- Models: `/mnt/c/ai_models`
- Datasets / runs: `/mnt/data`

Recommended env vars are provided in `.env.example`. At minimum, set:
- `EDGE_DEID_CACHE_HOME=/mnt/c/ai_cache`
- `EDGE_DEID_MODELS_HOME=/mnt/c/ai_models`
- `EDGE_DEID_DATA_HOME=/mnt/data`

GPU-focused example: `.env.gpu.example` (enables `USE_ONNX=true`, `USE_GPU=true`).

## 2) Install Dependencies

Option A (Conda):
```bash
conda env create -f env/conda.yaml
conda activate edgedeid
```

GPU-focused Conda env:
```bash
conda env create -f env/conda.gpu.yaml
conda activate ai_env
```

Option B (venv):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For ONNX Runtime GPU inference with venv/pip:
```bash
pip install -r requirements-gpu.txt
```

## 3) One-Time Downloads (Opt-in Network)

Download the base Chinese NER model locally:
```bash
PYTHONPATH=src python scripts/download_models.py --allow-network --only ner_zh
```

Download the base ONNX export (optional):
```bash
PYTHONPATH=src python scripts/download_models.py --allow-network --only ner_zh_onnx
```

If you plan to run ONNX inference on GPU, ensure your environment includes:
- CUDA-enabled PyTorch (for training)
- `onnxruntime-gpu` (for ONNX Runtime CUDAExecutionProvider)
- `onnxscript` (required by `torch.onnx.export` on newer PyTorch versions)

## 4) Run the Repeatable Multi-Dataset Training Pipeline

Profiles live under `configs/training/`. Example (first run, downloads datasets):
```bash
PYTHONPATH=src python scripts/run_multi_dataset_pipeline.py \
  --config configs/training/multi_zh_ner_demo.yaml \
  --allow-network \
  --trust-remote-code
```

Large PII + zh robustness profile (start with the smoke config, then scale up):
```bash
PYTHONPATH=src python scripts/run_multi_dataset_pipeline.py \
  --config configs/training/pii_large_mix_smoke_gpu.yaml \
  --allow-network \
  --trust-remote-code
```

Offline re-run (after caches and prepared JSONL exist):
```bash
PYTHONPATH=src python scripts/run_multi_dataset_pipeline.py \
  --config configs/training/multi_zh_ner_demo.yaml
```

Outputs:
- Prepared datasets: `/mnt/data/datasets/edge_deid/processed/<dataset_slug>/<split>/`
- Training runs: `/mnt/data/training/runs/edge_deid/<run_slug>/`
- Exported ONNX: `/mnt/c/ai_models/detection/edge_deid/<run_slug>/`
- Reports: `/mnt/data/training/logs/edge_deid/<run_name>/report.json` (from the profile)

Notebook equivalents:
- `notebooks/11_multi_dataset_pipeline.ipynb`
- `notebooks/12_pii_large_pipeline.ipynb`

## 5) GPT-2 Placeholder Fine-Tuning (Optional, Dev-only)

This workflow fine-tunes a Chinese GPT-2 style model on a **placeholder corpus** (e.g., `<NAME>`, `<EMAIL>`) for offline fake-data generation. It does not affect runtime de-identification unless you explicitly enable it.

Smoke run:
```bash
PYTHONPATH=src python scripts/run_gpt2_pipeline.py \
  --config configs/training/gpt2_zh_placeholder_smoke.yaml
```

Outputs:
- Corpus: `/mnt/data/datasets/edge_deid/processed/<run_slug>/<split>/corpus.jsonl`
- Training runs: `/mnt/data/training/runs/edge_deid/<run_slug>/`
- Models: `/mnt/c/ai_models/llm/edge_deid/<run_slug>/`

Notebook equivalent: `notebooks/13_gpt2_placeholder_pipeline.ipynb`

## 6) GPT-2 Rewriter Fine-Tuning (Optional, Dev-only)

This workflow fine-tunes a separate GPT-2 model to **rewrite de-identified output** into more fluent text (Model B). It trains on prompt-style pairs:

- INPUT: de-identified text with deterministic fake replacements + injected noise
- OUTPUT: the clean de-identified text (same replacement values)

Smoke run:
```bash
PYTHONPATH=src python scripts/run_gpt2_rewriter_pipeline.py \
  --config configs/training/gpt2_zh_rewriter_smoke.yaml
```

Outputs:
- Rewriter corpus: `/mnt/data/datasets/edge_deid/processed/<run_slug>/<split>/rewriter/corpus.jsonl`
- Training runs: `/mnt/data/training/runs/edge_deid/<run_slug>/`
- Models: `/mnt/c/ai_models/llm/edge_deid/<run_slug>/`

Notebook equivalent: `notebooks/14_gpt2_rewriter_pipeline.ipynb`

## 7) Iterative GPT-2 DeID Fine-Tuning (1-hour Loop, Dev-only)

This workflow fine-tunes a GPT-2 model to perform **de-identification + fake replacement + optional rewrite** using supervised pairs:

```json
{"id":"...", "input":"raw text (optionally with PII markup)", "output":"target de-identified text"}
```

Supported PII markup in `input`:
- XML: `<PII type="PHONE">0912-345-678</PII>`
- Brackets: `[PHONE]0912-345-678[/PHONE]`

If no markup is present, you can enable weak-label evaluation via regex rules (`configs/regex_zh.yaml`), but metrics will be labeled as a proxy.

Start a 1-hour iterative run (trains short rounds, evaluates every round, keeps top-K/last-K checkpoints):
```bash
PYTHONPATH=src python scripts/iterative_train.py \
  --config configs/training/deid_gpt2_iterative_zh.yaml
```

Outputs (default AI_WAREHOUSE layout):
- Prepared corpus: `/mnt/data/datasets/edge_deid/processed/<run_slug>/train/corpus.jsonl`
- Round checkpoints: `/mnt/data/training/runs/edge_deid/<run_slug>/rounds/round_*/`
- Evaluation artifacts per round: `/mnt/data/training/logs/edge_deid/<run_slug>/round_*/`
- Summary (append): `/mnt/data/training/logs/edge_deid/<run_slug>/summary.csv`
- Best pointer: `/mnt/data/training/logs/edge_deid/<run_slug>/best.json`
- Best model copy: `/mnt/c/ai_models/llm/edge_deid/<run_slug>/`

Notebook equivalent: `notebooks/15_iterative_deid_training.ipynb`
