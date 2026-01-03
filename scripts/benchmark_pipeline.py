from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout
from deid_pipeline import DeidPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the DeidPipeline end-to-end.")
    parser.add_argument("--input", default="", help="Optional input file path.")
    parser.add_argument("--chars", type=int, default=10_000, help="Synthetic text length for txt input.")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="Detection language.")
    parser.add_argument(
        "--mode",
        choices=["replace", "blackbox"],
        default="replace",
        help="Replacement mode.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs.")
    parser.add_argument("--output-dir", default="", help="Optional output directory for rebuilt artifacts.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    return parser.parse_args()


def synthetic_text(chars: int) -> str:
    base = (
        "ID A123456789 and phone 0912345678. "
        "Email test@example.com. Address \u53f0\u5317\u5e02\u4fe1\u7fa9\u8def1\u865f. "
    )
    chunks = []
    while sum(len(c) for c in chunks) < chars:
        chunks.append(base)
    return "".join(chunks)[:chars]


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

    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory()
        input_path = Path(temp_dir.name) / "bench.txt"
        input_path.write_text(synthetic_text(int(args.chars)), encoding="utf-8")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    pipeline = DeidPipeline(language=args.lang)

    for _ in range(int(args.warmup)):
        _ = pipeline.process(str(input_path), output_mode=args.mode, output_dir=output_dir)

    wall_ms: List[float] = []
    totals_ms: List[float] = []
    detects_ms: List[float] = []
    replaces_ms: List[float] = []
    entities_counts: List[int] = []

    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        result = pipeline.process(str(input_path), output_mode=args.mode, output_dir=output_dir)
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        totals_ms.append(float(result.timings_ms.get("total", 0.0)))
        detects_ms.append(float(result.timings_ms.get("detect", 0.0)))
        replaces_ms.append(float(result.timings_ms.get("replace", 0.0)))
        entities_counts.append(len(result.entities))

    report: Dict[str, Any] = {
        "input_path": str(input_path),
        "runs": int(args.runs),
        "warmup": int(args.warmup),
        "mode": args.mode,
        "lang": args.lang,
        "wall_ms": summarize_ms(wall_ms),
        "timings_ms_total": summarize_ms(totals_ms),
        "timings_ms_detect": summarize_ms(detects_ms),
        "timings_ms_replace": summarize_ms(replaces_ms),
        "entities_mean": float(sum(entities_counts) / len(entities_counts)) if entities_counts else 0.0,
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)

    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
