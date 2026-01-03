from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pipeline over a dataset directory and write a JSON report."
    )
    parser.add_argument(
        "--dataset-dir",
        default="advanced_test_dataset",
        help="Root directory containing input files.",
    )
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="Detection language.")
    parser.add_argument(
        "--mode",
        choices=["replace", "blackbox"],
        default="replace",
        help="Output mode.",
    )
    parser.add_argument("--output-json", default="pipeline_results.json", help="Output JSON report path.")
    return parser.parse_args()


def run_automated_test_pipeline(dataset_dir: Path, *, lang: str, mode: str) -> List[Dict[str, Any]]:
    from deid_pipeline import DeidPipeline

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    pipeline = DeidPipeline(language=lang)
    results: List[Dict[str, Any]] = []
    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            path = Path(root) / fn
            start = time.perf_counter()
            res = pipeline.process(str(path), output_mode=mode)
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "file": str(path),
                    "format": path.suffix.lstrip(".").lower(),
                    "pii_count": len(res.entities),
                    "processing_ms": elapsed * 1000.0,
                }
            )

    return results


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    results = run_automated_test_pipeline(Path(args.dataset_dir), lang=str(args.lang), mode=str(args.mode))

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
