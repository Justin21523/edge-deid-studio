# Dataset and License Notes (Training)

This repository is **offline-first** at runtime. Dataset ingestion and model training are **dev-only** workflows.

## Rules

- Do not commit real PII or proprietary documents.
- Do not commit raw third-party datasets into the repository.
- Always review the dataset license terms before downloading, using, or redistributing.
- Network downloads must be explicitly enabled in tooling (opt-in).

## Target Corpora

| Dataset | Intended use | Ingestion status | License notes |
|---|---|---|---|
| `tner/wikiann` (zh) | Base multilingual NER robustness | Supported via HF token NER adapter | Verify dataset card license before use |
| `levow/msra_ner` | Chinese NER robustness | Supported via HF token NER adapter | Verify dataset card license before use |
| `hltcoe/weibo_ner` | Chinese social NER robustness | Supported via HF token NER adapter | Verify dataset card license before use |
| `ai4privacy/pii-masking-300k` | PII masking / span extraction | Supported via masked-pair adapter (schema inference) | Must document and comply with license constraints prior to production use |
| `nvidia/Nemotron-PII` | PII masking / span extraction | Supported via masked-pair adapter (schema inference) | Must document and comply with license constraints prior to production use |

## How to Use Adapters

Adapters live in `src/deid_pipeline/training/datasets.py` and are network-gated by default.

- Recommended workflow:
  1) Prepare a local span-JSONL dataset with `scripts/prepare_dataset.py`.
  2) Train from the prepared JSONL via `scripts/train_token_classifier.py --input-jsonl ...`.

- Token-level BIO datasets:
  - Loaded as token/tag sequences and converted into span examples for training alignment.
- Masked-pair datasets:
  - Loaded as `(original_text, masked_text)` pairs and converted to spans by diffing placeholders (e.g., `<NAME>`).

If schema inference fails, update the adapter configuration with explicit field names and re-run.

## Prepared Dataset Output (Default)

Prepared datasets are written under the AI_WAREHOUSE layout by default:

- `/mnt/data/datasets/edge_deid/processed/<dataset_slug>/<split>/dataset.jsonl`
- `/mnt/data/datasets/edge_deid/processed/<dataset_slug>/<split>/manifest.json`
- `/mnt/data/datasets/edge_deid/processed/<dataset_slug>/<split>/quality.json`

You can also generate a standalone report for any prepared JSONL:

```bash
PYTHONPATH=src python scripts/report_dataset_quality.py \
  --input-jsonl /mnt/data/datasets/edge_deid/processed/<dataset_slug>/<split>/dataset.jsonl \
  --json-out /mnt/data/datasets/edge_deid/processed/<dataset_slug>/<split>/quality.report.json
```

## End-to-End Multi-Dataset Pipeline (Dev-only)

For a repeatable workflow that prepares multiple datasets, mixes them, fine-tunes a token-classifier, exports to ONNX, and benchmarks:
- Script: `scripts/run_multi_dataset_pipeline.py`
- Profiles: `configs/training/*.yaml`
- Notebook: `notebooks/11_multi_dataset_pipeline.ipynb`
