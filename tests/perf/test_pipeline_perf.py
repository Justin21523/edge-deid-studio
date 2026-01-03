from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from deid_pipeline import DeidPipeline


def _enabled() -> bool:
    return os.getenv("RUN_PERF_TESTS", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _enabled(), reason="Perf tests are disabled (set RUN_PERF_TESTS=1).")
def test_pipeline_text_p95_under_threshold(tmp_path: Path):
    threshold_ms = float(os.getenv("PIPELINE_TEXT_P95_MS", "150"))

    text = ("ID A123456789 and phone 0912345678. " * 400)[:10_000]
    input_path = tmp_path / "perf.txt"
    input_path.write_text(text, encoding="utf-8")

    pipeline = DeidPipeline(language="zh")

    warmup = int(os.getenv("PERF_WARMUP", "1"))
    runs = int(os.getenv("PERF_RUNS", "10"))

    for _ in range(warmup):
        _ = pipeline.process(str(input_path), output_mode="replace")

    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = pipeline.process(str(input_path), output_mode="replace")
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples = sorted(samples)
    p95 = samples[int(0.95 * (len(samples) - 1))]
    assert p95 <= threshold_ms, f"p95={p95:.2f}ms > threshold={threshold_ms:.2f}ms"
