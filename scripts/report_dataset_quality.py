from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout
from deid_pipeline.training.io import load_span_examples_jsonl
from deid_pipeline.training.quality import analyze_span_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a quality report for a prepared span-JSONL dataset.")
    parser.add_argument("--input-jsonl", required=True, help="Path to dataset.jsonl (text + entities).")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on loaded examples.")
    parser.add_argument("--max-issues", type=int, default=200, help="Max issues to include in the report.")
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    input_path = Path(args.input_jsonl).expanduser().resolve()
    examples = load_span_examples_jsonl(input_path)
    if int(args.max_examples) > 0:
        examples = examples[: int(args.max_examples)]

    report: Dict[str, Any] = analyze_span_examples(examples, max_issues=int(args.max_issues))

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

