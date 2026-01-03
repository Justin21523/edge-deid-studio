# Repository Guidelines

## Project Structure & Module Organization

- `src/deid_pipeline/`: core de-identification pipeline (handlers, detectors, replacement, runtime caches, training helpers).
- `configs/`: YAML configuration (for example `configs/regex_zh.yaml`).
- `scripts/`: dev-only utilities (dataset prep, training, ONNX export/quantization, benchmarks).
- `notebooks/`: notebook equivalents for data/training workflows (keep English-only).
- `tests/`: automated tests (primarily `pytest`), with opt-in perf tests under `tests/perf/`.
- `sensitive_data_generator/`: utilities for generating synthetic PII and multi-format test documents.
- `docs/`: design, usage, performance, and storage layout notes.

This repo assumes the **AI_WAREHOUSE 3.0** layout: models/caches under `/mnt/c`, datasets and training runs under `/mnt/data`.
See `docs/STORAGE_LAYOUT.md`. Repo-local `models/` and `edge_models/` are legacy fallbacks and are gitignored.

## Build, Test, and Development Commands

- Create an environment:
  - Conda: `conda env create -f env/conda.yaml && conda activate edgedeid` (Python 3.10)
  - venv: `python -m venv .venv && source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Make imports work with the `src/` layout (pick one):
  - Editable install: `pip install -e .`
  - Or: `export PYTHONPATH="$PWD/src"`
- Download/export local models (dev-only; requires internet): `python scripts/download_models.py --allow-network`
- Run the CLI: `python main.py -i test_input/clean-new.docx --mode replace --json --output-dir out`
- Quick smoke test: `python quick_tests.py --all`
- Run tests: `pytest -q`
- Prepare datasets (offline by default): `PYTHONPATH=src python scripts/prepare_dataset.py --dataset synthetic --split train`

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for functions/vars, `PascalCase` for classes.
- Prefer package-absolute imports (`from deid_pipeline...`) so code works when installed/editable.
- Keep configs in `configs/` (avoid hardcoding regex rules or thresholds in code).
- For large dev caches, prefer setting: `HF_HOME=/mnt/c/ai_cache/huggingface`, `TORCH_HOME=/mnt/c/ai_cache/torch`, `XDG_CACHE_HOME=/mnt/c/ai_cache`.

## Testing Guidelines

- Add/extend tests under `tests/` (name files `test_*.py` and tests `test_*`).
- Keep unit tests deterministic; use synthetic data from `sensitive_data_generator/` when possible.
- If a test depends on local sample files/models, skip gracefully when missing (see existing integration tests).

## Commit & Pull Request Guidelines

- Follow the existing Conventional Commit style seen in history: `feat(scope): ...`, `fix(scope): ...`, `refactor(scope): ...`, `chore: ...`, `ci: ...`.
- PRs should include: a short problem statement, testing steps (`pytest -q`), and sample output/screenshots for CLI/UI changes; link related issues.

## Security & Data Handling

- Do not commit real PII or customer documents—use synthetic fixtures and keep any sensitive material out of `test_input/`.
- Large model files and exported artifacts are intentionally ignored; prefer `scripts/download_models.py --allow-network` to fetch/rebuild them.
