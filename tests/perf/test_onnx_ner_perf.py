from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from deid_pipeline.config import Config
from deid_pipeline.pii.detectors.bert_onnx_detector import BertONNXNERDetector


def _enabled() -> bool:
    return os.getenv("RUN_PERF_TESTS", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _enabled(), reason="Perf tests are disabled (set RUN_PERF_TESTS=1).")
def test_onnx_session_p95_under_threshold():
    cfg = Config()
    model_path = Path(os.getenv("ONNX_MODEL_PATH", str(cfg.ONNX_MODEL_PATH))).expanduser().resolve()
    tokenizer_dir = Path(os.getenv("NER_MODEL_PATH_ZH", str(cfg.NER_MODEL_PATH_ZH))).expanduser().resolve()

    if not model_path.exists():
        pytest.skip(f"ONNX model not found: {model_path}")
    if not tokenizer_dir.exists():
        pytest.skip(f"Tokenizer dir not found: {tokenizer_dir}")

    threshold_ms = float(os.getenv("ONNX_NER_P95_MS", "25"))
    runs = int(os.getenv("PERF_RUNS", "20"))
    warmup = int(os.getenv("PERF_WARMUP", "3"))

    text = ("ID A123456789 and phone 0912345678. " * 400)[:10_000]
    detector = BertONNXNERDetector(model_path, tokenizer_dir, providers=list(cfg.ONNX_PROVIDERS))

    encoding = detector.tokenizer(
        text,
        return_tensors="np",
        return_offsets_mapping=True,
        truncation=True,
        max_length=detector.max_len,
        stride=detector.stride,
        return_overflowing_tokens=True,
        padding="max_length",
    )
    ort_inputs = {k: encoding[k] for k in detector.input_names if k in encoding}

    for _ in range(warmup):
        _ = detector.session.run([detector.output_name], ort_inputs)[0]

    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = detector.session.run([detector.output_name], ort_inputs)[0]
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples = sorted(samples)
    p95 = samples[int(0.95 * (len(samples) - 1))]
    assert p95 <= threshold_ms, f"p95={p95:.2f}ms > threshold={threshold_ms:.2f}ms"

