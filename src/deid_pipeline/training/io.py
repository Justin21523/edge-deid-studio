from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Dict, Iterable, List, Sequence

from .datasets import SpanExample


def dump_span_examples_jsonl(path: Path, examples: Iterable[SpanExample]) -> None:
    """Write span examples to JSONL (one example per line)."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row = {
                "text": ex.text,
                "entities": [
                    {
                        **dict(ent),
                        "span": list(ent.get("span", (0, 0))),
                    }
                    for ent in ex.entities
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_span_examples_jsonl(path: Path) -> Iterator[SpanExample]:
    """Iterate span examples from JSONL without loading the full file."""

    path = Path(path).expanduser().resolve()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            example = span_example_from_jsonl_line(line)
            if example is None:
                continue
            yield example


def load_span_examples_jsonl(path: Path) -> List[SpanExample]:
    """Load span examples from JSONL."""

    return list(iter_span_examples_jsonl(path))


def span_example_from_jsonl_line(line: str) -> SpanExample | None:
    """Parse a single JSONL line into a SpanExample."""

    raw = (line or "").strip()
    if not raw:
        return None

    row = json.loads(raw)
    text = str(row.get("text", "") or "")
    raw_entities = row.get("entities", []) or []
    entities = []
    for ent in raw_entities:
        span = ent.get("span")
        if span is not None:
            try:
                ent["span"] = (int(span[0]), int(span[1]))
            except Exception:
                pass
        entities.append(ent)
    return SpanExample(text=text, entities=tuple(entities))


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    """Write a small JSON manifest next to prepared dataset artifacts."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
