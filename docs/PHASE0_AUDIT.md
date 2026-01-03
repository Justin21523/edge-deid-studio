# Phase 0 Audit Report (EdgeDeID Studio)

Date: 2025-12-29

## Scope

This audit reviews the repository structure, runtime architecture, and the end-to-end de-identification flow with an emphasis on:
- Offline-first execution (no runtime network calls).
- Multi-format inputs and rebuild risks.
- Model/session initialization, caching, and performance hotspots.
- Contract stability (entity schema + result schema).

## Architecture Map

### Core runtime packages

- `src/deid_pipeline/__init__.py`
  - `DeidPipeline`: thin orchestrator (extract → detect → replace/mask → rebuild).
- `src/deid_pipeline/core/contracts.py`
  - Stable contracts: `Entity`, `DeidEvent`, `DeidResult`.
- `src/deid_pipeline/core/anchors.py`
  - Best-effort attachment of `page_index`/`bbox`/`cell` anchors from handler segments.
- `src/deid_pipeline/handlers/*`
  - Format handlers (`extract()` + `rebuild()`), normalized through `ExtractedDocument`.
- `src/deid_pipeline/pii/detectors/*`
  - Detector implementations (regex, HF token classification, ONNX token classification, spaCy legacy).
  - `CompositeDetector` resolves conflicts using `Config.ENTITY_PRIORITY`.
- `src/deid_pipeline/pii/utils/*`
  - Replacement (`Replacer`), fake data generation (`FakeProvider`).
- `src/deid_pipeline/runtime/*`
  - Cached model/tokenizer/session registries; ONNX provider selection.

### Dev-only training tooling

- `src/deid_pipeline/training/*`
  - Dataset adapters (network-gated), synthetic gold-span generator, token/span alignment utilities.
- `scripts/*`
  - Benchmarks, ONNX export/quantization, training entrypoints (dev-only).
- `notebooks/*`
  - English-only notebooks mirroring training/export/eval workflows.

## Critical Issues / Risks

1) **Format rebuild fidelity**
   - PDF/Image rebuild currently redacts at segment-level bounding boxes (coarse granularity).
   - Risk: over-redaction (entire block) or under-redaction (entity spans that cross segments).
   - Mitigation: add word-level OCR/PDF token bboxes and span-to-token alignment.

2) **Schema drift between detector entities and canonical entities**
   - Detector `Entity` (minimal) and canonical `core.contracts.Entity` (rich, multi-format) must remain compatible.
   - Risk: handlers/detectors add ad-hoc fields that are not normalized.
   - Mitigation: enforce normalization at pipeline boundary and add contract tests per detector/handler.

3) **Optional dependency surface area**
   - DOCX/XLSX/PPTX/PDF handling and model inference rely on optional packages.
   - Risk: inconsistent behavior across environments; silent downgrades.
   - Mitigation: explicit capability reporting in `artifacts` + consistent skip behavior in tests.

4) **Legacy entrypoints vs new pipeline**
   - `main.py` still uses legacy extraction APIs and does not use handler rebuild outputs.
   - Risk: users run a legacy path and miss rebuild/artifact improvements.
   - Mitigation: migrate CLI to `DeidPipeline.process(..., output_dir=...)` with compatibility flags.

5) **Performance realism**
   - ONNX perf targets (<25ms session time for 10k chars) are hardware-dependent and require a real local ONNX model.
   - Risk: regressions go unnoticed if perf tests are never enabled.
   - Mitigation: keep opt-in perf tests + publish benchmark procedure and reference machine baselines.

## Refactor Plan (Milestones)

### M1: Contract hardening
- Acceptance criteria:
  - `DeidResult` includes `entities`, `replacement_map`, `events`, `timings_ms`, `artifacts`.
  - Entity schema supports text spans + anchors (`page_index`, `bbox`, `cell`) and remains backward-compatible.
- Tests:
  - Unit tests for normalization and anchor attachment.

### M2: Handler completeness (multi-format rebuild)
- Acceptance criteria:
  - Handlers exist for TXT/HTML/CSV/PDF/Image/DOCX/XLSX/PPTX with consistent artifact outputs.
  - Rebuild is best-effort and capability is reported (`rebuild_supported`).
- Tests:
  - Unit tests for handler registry selection and CSV rebuild.
  - Integration tests for at least one binary format per environment where deps exist.

### M3: Edge inference hardening
- Acceptance criteria:
  - ONNX provider selection is centralized and filtered to available providers.
  - ONNX sessions are cached and initialized once per process.
- Perf checks:
  - Benchmarks produce JSON output and perf regression tests are opt-in.

### M4: Dataset + training tooling
- Acceptance criteria:
  - Synthetic gold-span dataset for CI exists and is deterministic.
  - Training/export/validation scripts exist and have notebook equivalents.
  - External dataset ingestion is explicit about license constraints and network gating.
- Tests:
  - Unit tests for synthetic generator + token/span alignment.

## Test Plan

- Unit tests (`tests/`): contracts, anchors, handlers, caches, training utilities.
- Integration tests (opt-in): format rebuild with optional dependencies.
- Perf tests (opt-in): `RUN_PERF_TESTS=1` for pipeline and ONNX session latency thresholds.

