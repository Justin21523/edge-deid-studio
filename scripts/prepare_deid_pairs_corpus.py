from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout
from deid_pipeline.training.prompts import PromptTemplate
from deid_pipeline.training.io import write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a DeID prompt+target corpus for causal-LM training (dev-only).")
    parser.add_argument("--input-jsonl", required=True, help="JSONL with {id,input,output}.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with {id,text}.")
    parser.add_argument(
        "--prompt-template",
        default="configs/prompts/deid_zh_v1.txt",
        help="Prompt template file path (must include {RAW_TEXT}).",
    )
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on processed examples.")
    parser.add_argument("--min-chars", type=int, default=20, help="Filter out very short examples.")
    parser.add_argument(
        "--split-paragraphs",
        action="store_true",
        help="If possible, split input/output by paragraph boundaries to reduce long examples.",
    )
    parser.add_argument("--manifest-out", default="", help="Optional manifest JSON output path.")
    return parser.parse_args()


def iter_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        for line in f:
            raw = (line or "").strip()
            if not raw:
                continue
            yield json.loads(raw)


def _paragraph_split(text: str) -> List[str]:
    # Keep it simple: split on double-newlines first, then single newlines.
    raw = str(text or "")
    if "\n\n" in raw:
        return [p for p in raw.split("\n\n") if p.strip()]
    if "\n" in raw:
        return [p for p in raw.split("\n") if p.strip()]
    return [raw]


def iter_pairs(row: Dict[str, Any], *, split_paragraphs: bool) -> Iterator[Tuple[str, str]]:
    inp = str(row.get("input", "") or "")
    out = str(row.get("output", "") or "")
    if not split_paragraphs:
        yield inp, out
        return

    in_parts = _paragraph_split(inp)
    out_parts = _paragraph_split(out)
    if len(in_parts) == len(out_parts) and len(in_parts) > 1:
        for a, b in zip(in_parts, out_parts):
            yield a, b
        return

    yield inp, out


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    layout = StorageLayout.from_project_root(repo_root)
    apply_cache_env_defaults(layout=layout)

    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = PromptTemplate.from_file(args.prompt_template)

    t0 = time.perf_counter()
    count_in = 0
    count_out = 0
    split_used = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in iter_rows(input_path):
            count_in += 1
            if int(args.max_examples) > 0 and count_in > int(args.max_examples):
                break

            ex_id = str(row.get("id", "") or row.get("_id", "") or row.get("example_id", "") or count_in)
            for idx, (inp, out) in enumerate(iter_pairs(row, split_paragraphs=bool(args.split_paragraphs))):
                if idx > 0:
                    split_used += 1
                rendered = prompt.render(inp)
                text = f"{rendered}{out}".strip()
                if int(args.min_chars) > 0 and len(text) < int(args.min_chars):
                    continue
                f.write(json.dumps({"id": f"{ex_id}:{idx}" if idx > 0 else ex_id, "text": text}, ensure_ascii=False) + "\n")
                count_out += 1

    load_ms = (time.perf_counter() - t0) * 1000.0

    manifest: Dict[str, Any] = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "prompt_template": str(Path(args.prompt_template).expanduser().resolve()),
        "max_examples": int(args.max_examples),
        "min_chars": int(args.min_chars),
        "split_paragraphs": bool(args.split_paragraphs),
        "counts": {"input": int(count_in), "output": int(count_out), "paragraph_splits": int(split_used)},
        "created_unix_s": time.time(),
        "load_ms": float(load_ms),
    }

    if args.manifest_out:
        write_manifest(Path(args.manifest_out), manifest)

    print(f"Wrote DeID LM corpus: {output_path} (count={count_out})")


if __name__ == "__main__":
    main()

