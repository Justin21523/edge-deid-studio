from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator

from deid_pipeline.pii.utils.fake_provider import FakeProvider
from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout
from deid_pipeline.training.io import write_manifest
from deid_pipeline.training.placeholders import canonicalize_placeholder_text, contains_cjk
from deid_pipeline.training.rewriter import (
    build_prompt,
    fill_placeholders_with_fake_values,
    hash16,
    make_noisy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a GPT-2 rewriter corpus (dev-only).")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL with a `text` field (placeholders).")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL with a `text` field (prompt+target).")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Language tag.")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on processed examples.")
    parser.add_argument("--min-chars", type=int, default=40, help="Filter out very short examples.")
    parser.add_argument("--seed", type=int, default=0, help="Noise seed.")
    parser.add_argument(
        "--filter-cjk",
        action="store_true",
        help="Keep only examples that contain CJK characters (recommended for zh).",
    )
    parser.add_argument(
        "--noise-punct-prob",
        type=float,
        default=0.35,
        help="Probability of swapping punctuation tokens.",
    )
    parser.add_argument(
        "--noise-space-prob",
        type=float,
        default=0.25,
        help="Probability of inserting extra spaces after punctuation.",
    )
    parser.add_argument(
        "--noise-dup-prob",
        type=float,
        default=0.08,
        help="Probability of duplicating punctuation.",
    )
    parser.add_argument("--manifest-out", default="", help="Optional JSON manifest output path.")
    return parser.parse_args()


def iter_text_rows(path: Path) -> Iterator[str]:
    """Iterate `text` rows from a generic JSONL file."""

    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        for line in f:
            raw = (line or "").strip()
            if not raw:
                continue
            row = json.loads(raw)
            text = str(row.get("text", "") or "")
            if text:
                yield text


def main() -> None:
    args = parse_args()
    layout = StorageLayout.from_project_root(Path(__file__).resolve().parents[1])
    apply_cache_env_defaults(layout=layout)

    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    provider = FakeProvider()
    language = str(args.language)
    filter_cjk = bool(args.filter_cjk) or language == "zh"

    t0 = time.perf_counter()
    count_in = 0
    count_out = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for text in iter_text_rows(input_path):
            count_in += 1
            if int(args.max_examples) > 0 and count_in > int(args.max_examples):
                break

            placeholder_text = canonicalize_placeholder_text(text)
            ctx = hash16(placeholder_text)
            clean = fill_placeholders_with_fake_values(placeholder_text, provider, context_hash=ctx)
            clean = clean.strip()
            if int(args.min_chars) > 0 and len(clean) < int(args.min_chars):
                continue
            if filter_cjk and not contains_cjk(clean):
                continue

            noisy = make_noisy(
                clean,
                rng,
                punct_prob=float(args.noise_punct_prob),
                space_prob=float(args.noise_space_prob),
                dup_prob=float(args.noise_dup_prob),
            )

            prompt = build_prompt(noisy, clean, language=language)
            out_f.write(json.dumps({"text": prompt}, ensure_ascii=False) + "\n")
            count_out += 1

    load_ms = (time.perf_counter() - t0) * 1000.0

    manifest: Dict[str, Any] = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "language": language,
        "filter_cjk": bool(filter_cjk),
        "max_examples": int(args.max_examples),
        "min_chars": int(args.min_chars),
        "noise": {
            "seed": int(args.seed),
            "punct_prob": float(args.noise_punct_prob),
            "space_prob": float(args.noise_space_prob),
            "dup_prob": float(args.noise_dup_prob),
        },
        "counts": {"input": int(count_in), "output": int(count_out)},
        "created_unix_s": time.time(),
        "load_ms": float(load_ms),
    }

    if args.manifest_out:
        out_path = Path(args.manifest_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(out_path, manifest)

    print(f"Wrote GPT-2 rewriter corpus: {output_path} (count={count_out})")


if __name__ == "__main__":
    main()
