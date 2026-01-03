from __future__ import annotations

import argparse
import json
import os
import time
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout, dataset_slug
from deid_pipeline.training.datasets import (
    HuggingFaceTokenNERAdapter,
    adapter_ai4privacy_pii_masking_300k,
    adapter_msra_ner,
    adapter_nemotron_pii,
    adapter_weibo_ner,
    adapter_wikiann,
    iter_token_examples_to_span_examples,
)
from deid_pipeline.training.placeholders import (
    canonicalize_placeholder_text,
    contains_cjk,
    replace_spans_with_placeholders,
)
from deid_pipeline.training.synthetic import generate_synthetic_span_examples
from deid_pipeline.training.io import write_manifest


def parse_dataset_spec(spec: str) -> Tuple[str, int]:
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("Empty dataset spec.")
    if ":" not in raw:
        return raw, 0
    name, max_raw = raw.rsplit(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid dataset spec: {spec}")
    try:
        max_examples = int(max_raw.strip())
    except Exception as exc:
        raise ValueError(f"Invalid max_examples in dataset spec: {spec}") from exc
    return name, int(max_examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local language-model corpus (dev-only).")
    parser.add_argument("--run-name", required=True, help="Stable name used for output directories.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Repeatable: `<dataset_name>[:max_examples]`.",
    )
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Language tag.")
    parser.add_argument("--split", default="train", help="Dataset split to load (default: train).")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Filter out very short examples.",
    )
    parser.add_argument(
        "--filter-cjk",
        action="store_true",
        help="Keep only texts that contain CJK characters (recommended for zh).",
    )
    parser.add_argument(
        "--no-filter-cjk",
        dest="filter_cjk",
        action="store_false",
        help="Disable CJK filtering.",
    )
    parser.add_argument(
        "--canonicalize-placeholders",
        dest="canonicalize_placeholders",
        action="store_true",
        help="Canonicalize placeholder tokens (e.g. <LASTNAME> -> <NAME>).",
    )
    parser.add_argument(
        "--no-canonicalize-placeholders",
        dest="canonicalize_placeholders",
        action="store_false",
        help="Disable placeholder canonicalization.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network access (required to download datasets).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom dataset loading code from the dataset repository (HF datasets).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory (defaults to /mnt/data/datasets/edge_deid/processed/<run_slug>/<split>).",
    )
    parser.set_defaults(filter_cjk=None, canonicalize_placeholders=True)
    return parser.parse_args()


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split())


def _iter_lm_texts(
    dataset_name: str,
    *,
    split: str,
    language: str,
    allow_network: bool,
    trust_remote_code: bool,
    max_examples: int = 0,
) -> Iterator[str]:
    name = str(dataset_name or "").strip()
    if not name:
        return

    if name == "synthetic":
        rows = generate_synthetic_span_examples(
            num_examples=int(max_examples) if int(max_examples) > 0 else 500,
            seed=0,
            language=language,
        )
        for row in rows:
            yield replace_spans_with_placeholders(str(row["text"]), list(row.get("entities", []) or ()))
        return

    if name in {"tner/wikiann", "levow/msra_ner", "hltcoe/weibo_ner"}:
        adapter = (
            adapter_wikiann(language=language)
            if name == "tner/wikiann"
            else adapter_msra_ner()
            if name == "levow/msra_ner"
            else adapter_weibo_ner()
        )
        token_examples = adapter.iter_load(
            split=split,
            allow_network=allow_network,
            trust_remote_code=trust_remote_code,
        )
        span_examples = iter_token_examples_to_span_examples(
            token_examples,
            separator="" if language == "zh" else " ",
            language=language,
            source=name,
        )
        for ex in span_examples:
            yield replace_spans_with_placeholders(ex.text, ex.entities)
        return

    if name in {"ai4privacy/pii-masking-300k", "nvidia/Nemotron-PII"}:
        adapter = (
            adapter_ai4privacy_pii_masking_300k(language=language)
            if name == "ai4privacy/pii-masking-300k"
            else adapter_nemotron_pii(language=language)
        )
        for text in adapter.iter_masked_texts(
            split=split,
            allow_network=allow_network,
            trust_remote_code=trust_remote_code,
        ):
            yield text
        return

    raise ValueError(f"Unsupported dataset for LM corpus: {name}")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    layout = StorageLayout.from_project_root(repo_root)
    apply_cache_env_defaults(layout=layout)

    run_slug = dataset_slug(str(args.run_name))

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = layout.edge_deid_datasets_home / "processed" / run_slug / str(args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = out_dir / "corpus.jsonl"
    allow_network = bool(args.allow_network)
    trust_remote_code = bool(args.trust_remote_code)
    language = str(args.language)
    split = str(args.split)

    filter_cjk = bool(args.filter_cjk) if args.filter_cjk is not None else language == "zh"
    canonicalize_placeholders = bool(getattr(args, "canonicalize_placeholders", True))
    min_chars = int(args.min_chars)

    dataset_specs = [parse_dataset_spec(s) for s in list(args.datasets or [])]
    if not dataset_specs:
        raise ValueError("No --dataset provided. Example: --dataset ai4privacy/pii-masking-300k:50000")

    t0 = time.perf_counter()
    sources: List[Dict[str, Any]] = []
    count = 0

    with corpus_path.open("w", encoding="utf-8") as f:
        for dataset_name, max_examples in dataset_specs:
            loaded = 0
            iterator = _iter_lm_texts(
                dataset_name,
                split=split,
                language=language,
                allow_network=allow_network,
                trust_remote_code=trust_remote_code,
                max_examples=int(max_examples),
            )
            if int(max_examples) > 0:
                iterator = islice(iterator, int(max_examples))

            for text in iterator:
                loaded += 1
                cleaned = _clean_text(text)
                if canonicalize_placeholders:
                    cleaned = canonicalize_placeholder_text(cleaned)
                if min_chars > 0 and len(cleaned) < min_chars:
                    continue
                if filter_cjk and not contains_cjk(cleaned):
                    continue
                f.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
                count += 1

            sources.append(
                {
                    "name": str(dataset_name),
                    "requested_max_examples": int(max_examples),
                    "loaded": int(loaded),
                }
            )

    load_ms = (time.perf_counter() - t0) * 1000.0

    manifest: Dict[str, Any] = {
        "run_name": str(args.run_name),
        "run_slug": run_slug,
        "language": language,
        "split": split,
        "sources": sources,
        "count": int(count),
        "min_chars": int(min_chars),
        "filter_cjk": bool(filter_cjk),
        "canonicalize_placeholders": bool(canonicalize_placeholders),
        "allow_network": bool(allow_network),
        "created_unix_s": time.time(),
        "load_ms": float(load_ms),
        "corpus_path": str(corpus_path),
    }
    write_manifest(out_dir / "manifest.json", manifest)

    print(f"Wrote LM corpus: {corpus_path} (count={count})")


if __name__ == "__main__":
    main()
