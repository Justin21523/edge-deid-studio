from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .datasets import SpanExample
from .io import dump_span_examples_jsonl, load_span_examples_jsonl, span_example_from_jsonl_line, write_manifest
from .quality import analyze_span_examples


@dataclass(frozen=True)
class SpanJsonlSource:
    """A prepared span-JSONL dataset source."""

    name: str
    jsonl_path: Path
    max_examples: int = 0


def load_span_jsonl_source(source: SpanJsonlSource) -> List[SpanExample]:
    """Load a prepared span-JSONL dataset, optionally truncating to `max_examples`."""

    examples = load_span_examples_jsonl(Path(source.jsonl_path).expanduser().resolve())
    if int(source.max_examples) > 0:
        examples = examples[: int(source.max_examples)]
    return examples


def mix_span_examples(
    sources: Sequence[Tuple[str, Sequence[SpanExample]]],
    *,
    shuffle: bool = True,
    seed: int = 0,
) -> List[SpanExample]:
    """Concatenate span examples from multiple sources and optionally shuffle."""

    combined: List[SpanExample] = []
    for _, examples in sources:
        combined.extend(list(examples))

    if shuffle and combined:
        rng = random.Random(int(seed))
        rng.shuffle(combined)

    return combined


def write_mixed_dataset(
    *,
    output_dir: Path,
    dataset_name: str,
    split: str,
    sources: Sequence[SpanJsonlSource],
    shuffle: bool = True,
    seed: int = 0,
    max_issues: int = 200,
) -> Dict[str, Any]:
    """Create a mixed dataset directory containing dataset.jsonl + manifest + quality."""

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_sources: List[Dict[str, Any]] = []

    jsonl_path = out_dir / "dataset.jsonl"
    quality_sample_size = 5000

    total_count = 0
    quality_examples: List[SpanExample] = []

    if shuffle:
        loaded_examples: List[Tuple[str, List[SpanExample]]] = []

        for src in sources:
            examples = load_span_jsonl_source(src)
            loaded_examples.append((src.name, examples))
            loaded_sources.append(
                {
                    "name": str(src.name),
                    "jsonl_path": str(Path(src.jsonl_path).expanduser().resolve()),
                    "max_examples": int(src.max_examples),
                    "loaded": int(len(examples)),
                }
            )

        combined = mix_span_examples(loaded_examples, shuffle=True, seed=int(seed))
        dump_span_examples_jsonl(jsonl_path, combined)
        total_count = int(len(combined))
        quality_examples = list(combined)
    else:
        rng = random.Random(int(seed))
        quality_sample_lines: List[str] = []

        with jsonl_path.open("w", encoding="utf-8") as out_f:
            for src in sources:
                src_path = Path(src.jsonl_path).expanduser().resolve()
                loaded = 0
                cap = int(src.max_examples)

                with src_path.open("r", encoding="utf-8") as in_f:
                    for line in in_f:
                        raw = (line or "").strip()
                        if not raw:
                            continue
                        if cap > 0 and loaded >= cap:
                            break

                        loaded += 1
                        total_count += 1
                        out_f.write(raw + "\n")

                        if quality_sample_size > 0:
                            if len(quality_sample_lines) < quality_sample_size:
                                quality_sample_lines.append(raw)
                            else:
                                j = rng.randint(0, total_count - 1)
                                if j < quality_sample_size:
                                    quality_sample_lines[j] = raw

                loaded_sources.append(
                    {
                        "name": str(src.name),
                        "jsonl_path": str(src_path),
                        "max_examples": int(src.max_examples),
                        "loaded": int(loaded),
                    }
                )

        for raw in quality_sample_lines:
            example = span_example_from_jsonl_line(raw)
            if example is not None:
                quality_examples.append(example)

    quality = analyze_span_examples(quality_examples, max_issues=int(max_issues))
    write_manifest(out_dir / "quality.json", quality)

    manifest: Dict[str, Any] = {
        "dataset": str(dataset_name),
        "split": str(split),
        "sources": loaded_sources,
        "shuffle": bool(shuffle),
        "seed": int(seed),
        "count": int(total_count),
        "quality_sample_count": int(len(quality_examples)),
        "created_unix_s": time.time(),
        "jsonl_path": str(jsonl_path),
    }
    write_manifest(out_dir / "manifest.json", manifest)

    return {
        "output_dir": str(out_dir),
        "jsonl_path": str(jsonl_path),
        "manifest_path": str(out_dir / "manifest.json"),
        "quality_path": str(out_dir / "quality.json"),
        "count": int(total_count),
    }
