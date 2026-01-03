from __future__ import annotations

from pathlib import Path

from deid_pipeline.training.datasets import SpanExample
from deid_pipeline.training.io import dump_span_examples_jsonl, load_span_examples_jsonl


def test_span_examples_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    examples = [
        SpanExample(
            text="ID A123456789.",
            entities=(
                {
                    "type": "ID",
                    "span": (3, 13),
                    "text": "A123456789",
                    "confidence": 1.0,
                    "score": 1.0,
                    "source": "test",
                    "language": "zh",
                },
            ),
        )
    ]

    dump_span_examples_jsonl(path, examples)
    loaded = load_span_examples_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].text == examples[0].text
    assert loaded[0].entities[0]["type"] == "ID"
    assert loaded[0].entities[0]["span"] == (3, 13)
