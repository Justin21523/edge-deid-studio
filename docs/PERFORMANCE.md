# Performance

This project targets edge-friendly latency while remaining offline-first.

## Targets

- Typical text workloads: `<150ms` end-to-end (10k chars, warm cache, regex + replacement).
- ONNX NER inference: `<25ms` p95 session time for 10k chars (model + hardware dependent).

## Benchmarks (Dev-only)

End-to-end pipeline benchmark:

```bash
python scripts/benchmark_pipeline.py --chars 10000 --runs 10 --lang zh
```

ONNX NER benchmark (total vs session-only):

```bash
python scripts/benchmark_onnx_ner.py \
  --onnx-model /mnt/c/ai_models/detection/edge_deid/bert-ner-zh.onnx \
  --tokenizer-dir /mnt/c/ai_models/detection/edge_deid/bert-ner-zh \
  --providers CUDAExecutionProvider,CPUExecutionProvider \
  --runs 20
```

Quantization (dynamic INT8 weights):

```bash
python scripts/quantize_onnx_model.py \
  --input /mnt/c/ai_models/detection/edge_deid/bert-ner-zh.onnx \
  --output /mnt/c/ai_models/detection/edge_deid/bert-ner-zh.int8.onnx
```

## Perf Regression Tests (Opt-in)

Perf tests are skipped by default to avoid flaky CI across machines:

```bash
RUN_PERF_TESTS=1 pytest -q
```

Environment overrides:
- `PIPELINE_TEXT_P95_MS` (default `150`)
- `ONNX_NER_P95_MS` (default `25`)
- `PERF_RUNS`, `PERF_WARMUP`

## Practical Notes

- Measure both cold-start and warm-cache runs (tokenizer/model/session caches).
- Ensure `USE_STUB=false` to measure real model latency.
- Keep model files local; runtime must not download models.
