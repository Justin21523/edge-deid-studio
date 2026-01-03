# EdgeDeID Studio

Offline-first, edge-optimized de-identification toolkit for multi-format documents.

## What This Repo Provides

- **Inputs**: `.txt`, `.html`, `.csv` (built-in); `.pdf`, images, `.docx`, `.xlsx`, `.pptx` (optional deps).
- **Pipeline**: extract → detect → resolve → replace/redact → rebuild (format-aware artifacts).
- **Detectors**:
  - Regex (always enabled as a recall backstop)
  - Hugging Face Transformers token-classification (local-only)
  - ONNX Runtime token-classification (local-only)
- **Replacement**: deterministic, cached per document context hash.
- **Privacy**: no network calls at runtime by default (downloads/training are dev-only scripts).

## Quick Start

### Install

```bash
pip install -e .
export PYTHONPATH="$PWD/src"
```

### Python API

```python
from deid_pipeline import DeidPipeline

result = DeidPipeline(language="zh").process(
    "example.csv",
    output_mode="replace",
    output_dir="out",
)

print(result.text)
print(result.entities)
print(result.artifacts.get("output_path"))
```

### CLI

```bash
python main.py -i test_input/sample.txt --mode replace --json --output-dir out
```

## Configuration (Environment Variables)

- `USE_STUB`: when `true`, disables heavy model detectors (defaults to `true`).
- `USE_ONNX`: enable ONNX detector if a local model exists (defaults to `false`).
- `ONNX_MODEL_PATH`: path to ONNX model (default: `/mnt/c/ai_models/detection/edge_deid/bert-ner-zh.onnx`).
- `NER_MODEL_PATH_ZH`: tokenizer/config directory for Chinese NER (default: `/mnt/c/ai_models/detection/edge_deid/bert-ner-zh`).
- `ONNX_PROVIDERS`: comma-separated ONNX Runtime providers (e.g. `CUDAExecutionProvider,CPUExecutionProvider`).
- `OCR_ENABLED`: enable OCR fallback for images/PDF scans (defaults to `true`).
- `USE_GPU`: enable GPU for OCR, and prefer GPU ONNX providers when available (defaults to `false`).
- `FAKER_LOCALE`: faker locale for replacements (default: `zh_TW`).

Cache locations (recommended for AI_WAREHOUSE 3.0):
- `HF_HOME=/mnt/c/ai_cache/huggingface`
- `TRANSFORMERS_CACHE=/mnt/c/ai_cache/huggingface`
- `TORCH_HOME=/mnt/c/ai_cache/torch`
- `XDG_CACHE_HOME=/mnt/c/ai_cache`

## Training (Dev-only)

Profiles live under `configs/training/` and can be run end-to-end (prepare → train → export → benchmark):

```bash
PYTHONPATH=src python scripts/run_multi_dataset_pipeline.py \
  --config configs/training/multi_zh_ner_demo.yaml \
  --allow-network \
  --trust-remote-code
```

See `docs/LOCAL_TRAINING.md` for setup and offline re-runs.

## Benchmarks

```bash
python scripts/benchmark_pipeline.py --chars 10000 --runs 10
python scripts/benchmark_onnx_ner.py --onnx-model /mnt/c/ai_models/detection/edge_deid/bert-ner-zh.onnx --tokenizer-dir /mnt/c/ai_models/detection/edge_deid/bert-ner-zh
python scripts/quantize_onnx_model.py --input /mnt/c/ai_models/detection/edge_deid/bert-ner-zh.onnx --output /mnt/c/ai_models/detection/edge_deid/bert-ner-zh.int8.onnx
```

Perf regression tests (opt-in):

```bash
RUN_PERF_TESTS=1 pytest -q
```

## Repository Layout

- `src/deid_pipeline/`: pipeline, contracts, detectors, replacer, and runtime caches.
- `src/deid_pipeline/handlers/`: format handlers (extract + rebuild).
- `sensitive_data_generator/`: synthetic data and multi-format test document generation.
- `scripts/`: developer utilities (benchmarks, model export/quantization).
- `tests/`: unit tests (+ opt-in perf tests under `tests/perf/`).

## Contributing

See `AGENTS.md`.

## More Docs

- `docs/USAGE.md`
- `docs/PERFORMANCE.md`
- `docs/STORAGE_LAYOUT.md`
- `docs/LOCAL_TRAINING.md`
