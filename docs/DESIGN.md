# Design Doc (Phase 1–4): EdgeDeID Studio

## Goals

- Provide a production-grade, modular de-identification toolkit optimized for edge execution.
- Support multi-format inputs (PDF, images, CSV/Excel, DOCX, PPTX where supported).
- Enforce stable data contracts for entities and results.
- Ensure deterministic, cached replacements with locale support (default `zh_TW`).
- Keep runtime offline-first (no network calls unless explicitly enabled for dev).

## Non-Goals (for this milestone)

- Perfect layout-preserving rebuild for every binary format (PDF/DOCX/XLSX/PPTX) under all edge cases.
- Training pipelines that require network access by default.
- Cloud services integration.

## Core Contracts

### Entity

Canonical entity schema lives in `src/deid_pipeline/core/contracts.py`:
- Text anchor: `span=(start,end)` in extracted text coordinates, plus best-effort `text`.
- Multi-format anchors (optional): `page_index`, `bbox` (PDF/image), `cell` (tables).
- Provenance: `source`, `language`, `confidence` (`score` retained as legacy alias).

### DeidResult

`DeidResult` is returned by `DeidPipeline.process()`:
- `entities`: normalized entities (canonical schema).
- `text`: replaced/masked output text (document-level concatenation).
- `replacement_map`: deterministic mapping keyed by `entity_type:original` (Phase 1 key).
- `events`: replacement audit events.
- `timings_ms`: stage timings.
- `artifacts`: rebuild outputs (e.g., `output_path`, `visual_result`), plus metadata.

## Pipeline Architecture

`src/deid_pipeline/__init__.py` implements `DeidPipeline` as a thin orchestrator:

1) Select handler by extension (`handlers/registry.py`).
2) `extract()` returns an `ExtractedDocument`:
   - `text`: handler-defined concatenation.
   - `segments`: slices of text with optional anchors.
3) `detect()` runs a `CompositeDetector` (regex + optional model detectors).
4) `replace()` performs deterministic replacement/masking with caching.
5) `rebuild()` delegates to the handler to write artifacts (optional `output_dir`).

## Format Handlers

Handlers implement `FormatHandler` (`src/deid_pipeline/handlers/base.py`):
- `extract(input_path, language) -> ExtractedDocument`
- `rebuild(document, output_text, entities, replacement_map, events, output_dir, mode) -> artifacts`

### Segments and Anchors

Handlers emit `SegmentSpec` values which become `TextSegment` entries:
- `start/end` are computed in concatenated `document.text`.
- Optional anchors: `page_index`, `bbox`, `cell`.

Pipeline attaches segment anchors to entities (`core/anchors.py`) when a span is fully contained in a segment.

## Detector Framework

- `RegexDetector` loads YAML rules and supports hot reload.
- `BertNERDetector` uses cached HF tokenizer/model (local-only).
- `BertONNXNERDetector` uses cached ONNX Runtime session and local tokenizer/config.
- `CompositeDetector` merges and resolves overlaps via `Config.ENTITY_PRIORITY`.

### Conflict Resolution

Overlapping spans are resolved with:
1) Type priority (`ENTITY_PRIORITY`)
2) Confidence score
3) Stable tie-breakers

## Fake Data and Replacement

- `Replacer` performs in-place replacement from the end of the string to preserve offsets.
- Deterministic cache key: `(entity_type, original, document_context_hash)`.
- `FakeProvider.generate_deterministic()` uses a stable seed derived from SHA-256.

Future extension (optional):
- Contextual generation (e.g., LLM/GPT-2) remains dev-only and must be local-only.

## Edge Inference (ONNX)

`src/deid_pipeline/runtime/onnx.py` centralizes:
- Provider filtering (`select_onnx_providers`)
- Session options (`create_session_options`)

`BertONNXNERDetector` performs:
- Tokenization (`return_tensors="np"`)
- ONNX session inference
- NumPy softmax + BIO decoding + chunk merge

Quantization:
- `scripts/quantize_onnx_model.py` performs dynamic INT8 weight quantization when `onnx` is installed.

## Benchmarks and Perf Regression

- `scripts/benchmark_pipeline.py`: end-to-end pipeline timing (JSON output).
- `scripts/benchmark_onnx_ner.py`: ONNX detector timing (total + session-only).
- Perf tests under `tests/perf/` are opt-in via `RUN_PERF_TESTS=1`.

## Dataset Strategy and Training Tooling

Dev-only training lives under `src/deid_pipeline/training/`:
- External dataset adapters (network-gated).
- Deterministic synthetic dataset generator with gold spans for CI.
- Token/span alignment helpers.

Entry points (dev-only):
- `scripts/train_token_classifier.py`
- `scripts/export_token_classifier_onnx.py`
- `scripts/validate_onnx_token_classifier.py`

Notebook equivalents live in `notebooks/` and are English-only.

## Security and Configuration

- Runtime must remain offline-first.
- All downloads/training must be explicitly enabled and must not run in production by default.
- Never commit real PII or customer documents; use synthetic fixtures.

