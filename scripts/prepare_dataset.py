from __future__ import annotations

import argparse
import json
import random
import time
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List

from deid_pipeline.storage.env import apply_cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout, dataset_slug
from deid_pipeline.training.datasets import (
    HuggingFaceTokenNERAdapter,
    SpanExample,
    adapter_ai4privacy_pii_masking_300k,
    adapter_nemotron_pii,
    iter_token_examples_to_span_examples,
)
from deid_pipeline.training.io import write_manifest
from deid_pipeline.training.quality import analyze_span_examples
from deid_pipeline.training.synthetic import generate_synthetic_span_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a dataset into span-JSONL format (dev-only).")
    parser.add_argument(
        "--dataset",
        default="synthetic",
        help=(
            "Source dataset. Supported: synthetic, tner/wikiann, levow/msra_ner, "
            "hltcoe/weibo_ner, ai4privacy/pii-masking-300k, nvidia/Nemotron-PII"
        ),
    )
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Language tag for output examples.")
    parser.add_argument("--dataset-config", default="", help="Optional dataset config name (e.g. 'zh').")
    parser.add_argument("--split", default="train", help="Dataset split to load (default: train).")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on loaded examples.")
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
        help="Optional output directory (defaults to /mnt/data/datasets/edge_deid/processed/<dataset>/<split>).",
    )
    return parser.parse_args()


def _reservoir_sample(
    rng: random.Random,
    *,
    sample: List[SpanExample],
    seen: int,
    item: SpanExample,
    k: int,
) -> None:
    if k <= 0:
        return
    if len(sample) < k:
        sample.append(item)
        return
    j = rng.randint(0, seen)
    if j < k:
        sample[j] = item


def _iter_span_examples(args: argparse.Namespace) -> Iterable[SpanExample]:
    dataset_name = str(args.dataset or "synthetic").strip()
    max_examples = int(args.max_examples)

    if dataset_name == "synthetic":
        rows = generate_synthetic_span_examples(
            num_examples=max_examples if max_examples > 0 else 500,
            seed=0,
            language=args.language,
        )
        for row in rows:
            yield SpanExample(text=row["text"], entities=tuple(row["entities"]))
        return

    if dataset_name in {"tner/wikiann", "levow/msra_ner", "hltcoe/weibo_ner"}:
        cfg = str(args.dataset_config).strip() or (
            args.language
            if dataset_name == "tner/wikiann"
            else "msra_ner"
            if dataset_name == "levow/msra_ner"
            else "default"
            if dataset_name == "hltcoe/weibo_ner"
            else None
        )
        adapter = HuggingFaceTokenNERAdapter(dataset_name, config_name=cfg)
        token_examples = adapter.iter_load(
            split=str(args.split),
            allow_network=bool(args.allow_network),
            trust_remote_code=bool(args.trust_remote_code),
        )
        span_examples = iter_token_examples_to_span_examples(
            token_examples,
            separator="" if args.language == "zh" else " ",
            language=args.language,
            source=dataset_name,
        )
        if max_examples > 0:
            yield from islice(span_examples, max_examples)
        else:
            yield from span_examples
        return

    if dataset_name in {"ai4privacy/pii-masking-300k", "nvidia/Nemotron-PII"}:
        adapter = (
            adapter_ai4privacy_pii_masking_300k(language=args.language)
            if dataset_name == "ai4privacy/pii-masking-300k"
            else adapter_nemotron_pii(language=args.language)
        )
        span_examples = adapter.iter_span_examples(
            split=str(args.split),
            allow_network=bool(args.allow_network),
            trust_remote_code=bool(args.trust_remote_code),
        )
        if max_examples > 0:
            yield from islice(span_examples, max_examples)
        else:
            yield from span_examples
        return

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    layout = StorageLayout.from_project_root(repo_root)
    apply_cache_env_defaults(layout=layout)

    slug = dataset_slug(str(args.dataset))

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = layout.edge_deid_datasets_home / "processed" / slug / str(args.split)

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rng = random.Random(0)
    quality_sample_size = 5000
    quality_sample: List[SpanExample] = []

    jsonl_path = out_dir / "dataset.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with jsonl_path.open("w", encoding="utf-8") as f:
        for count, ex in enumerate(_iter_span_examples(args), start=1):
            _reservoir_sample(rng, sample=quality_sample, seen=count - 1, item=ex, k=quality_sample_size)
            row = {
                "text": ex.text,
                "entities": [
                    {**dict(ent), "span": list(ent.get("span", (0, 0)))} for ent in ex.entities
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    load_ms = (time.perf_counter() - t0) * 1000.0

    manifest: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "dataset_slug": slug,
        "dataset_config": str(args.dataset_config or ""),
        "split": str(args.split),
        "language": str(args.language),
        "requested_max_examples": int(args.max_examples),
        "count": int(count),
        "allow_network": bool(args.allow_network),
        "created_unix_s": time.time(),
        "load_ms": float(load_ms),
        "quality_sample_size": int(quality_sample_size),
        "quality_sample_count": int(len(quality_sample)),
        "jsonl_path": str(jsonl_path),
    }
    write_manifest(out_dir / "manifest.json", manifest)

    quality = analyze_span_examples(quality_sample, max_issues=200)
    write_manifest(out_dir / "quality.json", quality)

    print(f"Wrote prepared dataset: {jsonl_path} (count={count})")


if __name__ == "__main__":
    main()
