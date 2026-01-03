# GitHub Issues Style Task Board

This board is organized as **Epic → Stories → Tasks**. Each task includes a definition of done.

## Epic 1) Core Contracts & Entity Schema

### Story: Stabilize public contracts

- [x] Task: Define canonical `Entity` and `DeidResult`
  - Goal: Provide stable, multi-format contracts used across the pipeline.
  - Files to touch: `src/deid_pipeline/core/contracts.py`
  - Implementation notes: Keep backward-compatible aliases (`score` for `confidence`).
  - Acceptance criteria:
    - `Entity` supports `span`, `page_index`, `bbox`, `cell`, `confidence`, `source`, `language`.
    - `DeidResult` includes entities, replacement map, events, timings, artifacts.
  - Tests: `tests/test_contracts.py`
  - Perf checks: N/A

- [x] Task: Add segment-to-entity anchor attachment
  - Goal: Attach `page_index`/`bbox`/`cell` when spans map to extracted segments.
  - Files to touch: `src/deid_pipeline/core/anchors.py`, `src/deid_pipeline/__init__.py`
  - Implementation notes: Best-effort mapping; only attach when fully contained in one segment.
  - Acceptance criteria:
    - Anchors appear in entities for segment-contained spans.
    - Cross-segment spans do not get incorrect anchors.
  - Tests: `tests/test_anchors.py`
  - Perf checks: O(n log m) max, no measurable regression.

## Epic 2) Format Handlers (PDF/Image/DOCX/XLSX/PPTX)

### Story: Handler interface and registry

- [x] Task: Introduce handler registry and base contracts
  - Goal: Make format handling pluggable and isolate extraction/rebuild logic.
  - Files to touch: `src/deid_pipeline/handlers/base.py`, `src/deid_pipeline/handlers/registry.py`
  - Implementation notes: Use `ExtractedDocument` + segment specs as the normalized representation.
  - Acceptance criteria:
    - Registry selects handlers by extension.
    - Handlers implement `extract()` and `rebuild()` signatures.
  - Tests: `tests/test_handlers.py`
  - Perf checks: N/A

### Story: Multi-format extraction and rebuild

- [x] Task: Add CSV handler with cell anchors + rebuild
  - Goal: Support structured tabular inputs with cell-level anchors.
  - Files to touch: `src/deid_pipeline/handlers/csv.py`, `src/deid_pipeline/handlers/registry.py`
  - Implementation notes: Represent each cell as a segment with `cell={row,col,address}`.
  - Acceptance criteria:
    - `output_dir` produces `*.deid.csv` with replacements applied.
  - Tests: `tests/test_csv_handler.py`
  - Perf checks: CSV rebuild scales linearly with cells.

- [x] Task: Improve PDF redaction to word-level bboxes
  - Goal: Avoid block-level over-redaction by narrowing redaction rectangles to matched text.
  - Files to touch: `src/deid_pipeline/handlers/pdf.py`
  - Implementation notes:
    - Use `page.search_for(entity.text)` and filter matches that intersect the segment bbox.
    - Fall back to segment bbox when text search fails.
  - Acceptance criteria:
    - Redaction is limited to entity text when text search succeeds.
    - Safe fallback behavior when text search fails.
  - Tests: `tests/test_pdf_handler_redaction.py` (skips if PyMuPDF missing)
  - Perf checks: Redaction overhead < 20ms for typical 1–3 page PDFs.

- [x] Task: Improve image redaction to token-level OCR boxes
  - Goal: Avoid redacting the entire OCR block when only part matches.
  - Files to touch: `src/deid_pipeline/handlers/image.py`
  - Implementation notes:
    - Use handler segments as token boxes and redact segments that overlap the entity span.
    - Fall back to `entity.bbox` when present.
  - Acceptance criteria:
    - Multi-token spans redact multiple tight boxes (best-effort).
    - Behavior is deterministic given fixed segment bboxes.
  - Tests: `tests/test_image_handler_redaction.py`
  - Perf checks: OCR must remain singleton and cached.

## Epic 3) Detector Framework + Conflict Resolver

### Story: Detector composition and caching

- [x] Task: Cache HF models/tokenizers and ONNX sessions
  - Goal: Avoid repeated initialization and improve latency.
  - Files to touch: `src/deid_pipeline/runtime/registry.py`
  - Implementation notes: Use `lru_cache` + local-only model loading.
  - Acceptance criteria: Multiple pipeline calls do not re-initialize models.
  - Tests: Covered indirectly via unit tests; add explicit cache tests if regressions occur.
  - Perf checks: Verify warm vs cold start in benchmarks.

- [x] Task: Centralize ONNX provider selection
  - Goal: Ensure stable provider selection across machines and avoid invalid providers.
  - Files to touch: `src/deid_pipeline/runtime/onnx.py`, `src/deid_pipeline/runtime/registry.py`
  - Implementation notes: Filter requested providers against `ort.get_available_providers()`.
  - Acceptance criteria: Session initialization succeeds with CPU-only environments.
  - Tests: `tests/test_onnx_runtime.py`
  - Perf checks: Provider filtering must be negligible (<1ms).

- [x] Task: Make spaCy detector use cached pipelines
  - Goal: Prevent repeated `spacy.load()` calls across runs.
  - Files to touch: `src/deid_pipeline/pii/detectors/legacy/spacy_detector.py`, `src/deid_pipeline/runtime/registry.py`
  - Implementation notes: Route through `get_spacy_pipeline()` and inject EntityRuler only once.
  - Acceptance criteria: spaCy pipeline loads once per process.
  - Tests: `tests/test_spacy_detector_cache.py` (skips if spaCy missing)
  - Perf checks: No regression in cold-start path when spaCy disabled.

## Epic 4) Fake Data Provider (deterministic + locale-aware + caching)

### Story: Determinism and stability

- [x] Task: Deterministic replacement cache keyed by `(type, original, context_hash)`
  - Goal: Ensure stable replacements within a document and across runs.
  - Files to touch: `src/deid_pipeline/replace/cache.py`, `src/deid_pipeline/pii/utils/replacer.py`
  - Implementation notes: Use a small thread-safe LRU.
  - Acceptance criteria: Same input triple produces identical replacement values.
  - Tests: `tests/test_replacement_cache.py`
  - Perf checks: Replacement overhead < 5ms for 10k chars.

- [x] Task: Add locale-aware typed generators (zh_TW)
  - Goal: Improve realism for names/addresses/phones and keep determinism.
  - Files to touch: `src/deid_pipeline/pii/utils/fake_provider.py`
  - Implementation notes: Use `faker` when installed; provide deterministic fallback.
  - Acceptance criteria:
    - Deterministic replacements per context hash.
    - No network calls; local-only.
  - Tests: `tests/test_fake_provider.py` (covers Faker fallback via monkeypatch)
  - Perf checks: Provider init is O(1) and cached.

## Epic 5) Benchmark & Profiling

### Story: Benchmark scripts and JSON outputs

- [x] Task: Add pipeline and ONNX benchmarks
  - Goal: Provide repeatable timing metrics and JSON outputs for regression tracking.
  - Files to touch: `scripts/benchmark_pipeline.py`, `scripts/benchmark_onnx_ner.py`
  - Implementation notes: Separate tokenization vs session timing for ONNX.
  - Acceptance criteria: Scripts print valid JSON and support `--json-out`.
  - Tests: N/A (scripts); covered by perf tests.
  - Perf checks: Benchmarks must not mutate global state beyond caches.

- [x] Task: Add opt-in perf regression tests
  - Goal: Catch latency regressions without flaking CI by default.
  - Files to touch: `tests/perf/test_pipeline_perf.py`, `tests/perf/test_onnx_ner_perf.py`
  - Implementation notes: Gate with `RUN_PERF_TESTS=1` and env thresholds.
  - Acceptance criteria: Tests are skipped by default; runnable locally on demand.
  - Tests: `pytest -q` (with/without env var)
  - Perf checks: Thresholds configurable via env vars.

## Epic 6) Test Pyramid (unit/integration/e2e/perf)

### Story: Increase coverage without relying on optional deps

- [x] Task: Add integration fixtures for PDF/DOCX/XLSX/PPTX
  - Goal: Validate rebuild behavior end-to-end when optional deps are present.
  - Files to touch: `tests/test_format_rebuilds.py`, `tests/test_pdf_handler_redaction.py`
  - Implementation notes: Generate synthetic documents on the fly; skip gracefully if deps missing.
  - Acceptance criteria: Integration suite passes on fully provisioned dev env.
  - Tests: `pytest -q` (tests are skipped if optional deps are missing)
  - Perf checks: N/A

## Epic 7) Optional: Training & ONNX Export Tooling

### Story: Offline-first training scaffolding + notebooks

- [x] Task: Add synthetic gold-span dataset generator for CI
  - Goal: Provide deterministic examples with gold spans for token alignment testing.
  - Files to touch: `src/deid_pipeline/training/synthetic.py`
  - Implementation notes: Seeded RNG; outputs `text` + `entities`.
  - Acceptance criteria: Same seed produces identical examples.
  - Tests: `tests/test_training_utils.py`
  - Perf checks: Generation < 50ms for 1k examples.

- [x] Task: Implement dataset adapters for external corpora
  - Goal: Ingest WikiAnn/MSRA/Weibo and PII masking datasets with explicit license notes.
  - Files to touch: `src/deid_pipeline/training/datasets.py`, `docs/DATASETS.md`
  - Implementation notes: Network must be explicitly enabled; prefer local paths.
  - Acceptance criteria: Adapters load and normalize tags/types into canonical schema.
  - Tests: Unit tests for normalization and schema inference (no network).
  - Perf checks: Adapter conversion is linear in dataset size.

- [x] Task: Provide training/export scripts + notebook equivalents
  - Goal: Support token classification training → ONNX export → validation.
  - Files to touch: `scripts/train_token_classifier.py`, `scripts/export_token_classifier_onnx.py`, `scripts/validate_onnx_token_classifier.py`, `notebooks/*`
  - Implementation notes: Notebooks should call scripts for reproducibility.
  - Acceptance criteria: Scripts run in a dev environment with required deps.
  - Tests: Smoke test via notebooks; unit tests cover helpers.
  - Perf checks: N/A

- [x] Task: Add dataset preparation script + JSONL cache
  - Goal: Produce offline-ready span-JSONL datasets stored under `/mnt/data` by default.
  - Files to touch: `scripts/prepare_dataset.py`, `src/deid_pipeline/training/io.py`, `notebooks/08_prepare_datasets.ipynb`
  - Implementation notes:
    - Must be network-gated by default (`--allow-network`).
    - Write `dataset.jsonl` + `manifest.json` for reproducibility.
  - Acceptance criteria:
    - Synthetic dataset preparation works without network.
    - Prepared JSONL can be consumed via `scripts/train_token_classifier.py --input-jsonl ...`.
  - Tests: `tests/test_training_io.py`
  - Perf checks: Linear in number of examples.
