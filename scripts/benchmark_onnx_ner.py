from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout
from deid_pipeline.pii.detectors.bert_onnx_detector import BertONNXNERDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX NER detector (offline/local-only).")
    parser.add_argument("--onnx-model", required=True, help="Path to ONNX token-classification model.")
    parser.add_argument("--tokenizer-dir", required=True, help="Path to local tokenizer/config directory.")
    parser.add_argument(
        "--providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
        help="Comma-separated provider list (filtered to available providers).",
    )
    parser.add_argument("--chars", type=int, default=10_000, help="Synthetic text length.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs.")
    parser.add_argument("--runs", type=int, default=10, help="Measured runs.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    return parser.parse_args()


def synthetic_text(chars: int) -> str:
    base = (
        "Patient ID A123456789 visited on 2025-01-01. "
        "Phone 0912345678, email test@example.com. "
        "Address \u53f0\u5317\u5e02\u4fe1\u7fa9\u8def1\u865f. "
    )
    chunks = []
    while sum(len(c) for c in chunks) < chars:
        chunks.append(base)
    text = "".join(chunks)
    return text[:chars]


def summarize_ms(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    sorted_samples = sorted(samples)
    p50 = statistics.median(sorted_samples)
    p95 = sorted_samples[int(0.95 * (len(sorted_samples) - 1))]
    mean = sum(sorted_samples) / len(sorted_samples)
    return {"p50": float(p50), "p95": float(p95), "mean": float(mean)}


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    onnx_path = Path(args.onnx_model).expanduser().resolve()
    tok_dir = Path(args.tokenizer_dir).expanduser().resolve()
    providers = tuple(p.strip() for p in args.providers.split(",") if p.strip())

    detector = BertONNXNERDetector(onnx_path, tok_dir, providers=list(providers))
    text = synthetic_text(int(args.chars))

    for _ in range(int(args.warmup)):
        _ = detector.detect(text)

    total_ms: List[float] = []
    entity_counts: List[int] = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        entities = detector.detect(text)
        total_ms.append((time.perf_counter() - t0) * 1000.0)
        entity_counts.append(len(entities))

    # Session-only timing (tokenization excluded).
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

    session_ms: List[float] = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        _ = detector.session.run([detector.output_name], ort_inputs)[0]
        session_ms.append((time.perf_counter() - t0) * 1000.0)

    report: Dict[str, Any] = {
        "chars": len(text),
        "runs": int(args.runs),
        "warmup": int(args.warmup),
        "chunks": int(encoding["input_ids"].shape[0]),
        "providers_requested": list(providers),
        "providers_used": detector.session.get_providers(),
        "detect_total_ms": summarize_ms(total_ms),
        "onnx_session_ms": summarize_ms(session_ms),
        "entities_mean": float(sum(entity_counts) / len(entity_counts)) if entity_counts else 0.0,
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
