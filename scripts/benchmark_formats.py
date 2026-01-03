from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DeidPipeline runtime across multiple file formats."
    )
    parser.add_argument(
        "--dataset-dir",
        default="advanced_test_dataset",
        help="Directory containing sample documents.",
    )
    parser.add_argument(
        "--extensions",
        default="pdf,docx,xlsx,png",
        help="Comma-separated file extensions (without dots).",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max files per extension.")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="Detection language.")
    parser.add_argument(
        "--mode",
        choices=["replace", "blackbox"],
        default="replace",
        help="Output mode.",
    )
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    return parser.parse_args()


def benchmark_formats(
    dataset_dir: Path,
    extensions: List[str],
    *,
    limit: int,
    lang: str,
    mode: str,
) -> Dict[str, Any]:
    from deid_pipeline import DeidPipeline

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    pipeline = DeidPipeline(language=lang)
    results: Dict[str, Any] = {"dataset_dir": str(dataset_dir), "formats": {}}

    for ext in extensions:
        ext_clean = ext.lower().lstrip(".")
        files = sorted(
            p
            for p in dataset_dir.iterdir()
            if p.is_file() and p.suffix.lower() == f".{ext_clean}"
        )

        if not files:
            results["formats"][ext_clean] = {"count": 0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
            continue

        samples_ms: List[float] = []
        for path in files[: max(0, int(limit))]:
            t0 = time.perf_counter()
            _ = pipeline.process(str(path), output_mode=mode)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)

        results["formats"][ext_clean] = {
            "count": len(samples_ms),
            "avg_ms": float(sum(samples_ms) / len(samples_ms)),
            "min_ms": float(min(samples_ms)),
            "max_ms": float(max(samples_ms)),
        }

    return results


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    extensions = [s.strip() for s in str(args.extensions).split(",") if s.strip()]
    report = benchmark_formats(
        Path(args.dataset_dir),
        extensions,
        limit=int(args.limit),
        lang=str(args.lang),
        mode=str(args.mode),
    )

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
