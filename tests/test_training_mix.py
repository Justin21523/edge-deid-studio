from __future__ import annotations

import json
import random
from pathlib import Path

from deid_pipeline.training.datasets import SpanExample
from deid_pipeline.training.io import dump_span_examples_jsonl, load_span_examples_jsonl
from deid_pipeline.training.mix import SpanJsonlSource, mix_span_examples, write_mixed_dataset


def _span_example(
    text: str,
    span: tuple[int, int],
    *,
    entity_type: str,
    source: str,
    language: str = "zh",
) -> SpanExample:
    start, end = span
    return SpanExample(
        text=text,
        entities=(
            {
                "type": entity_type,
                "span": (start, end),
                "text": text[start:end],
                "confidence": 1.0,
                "score": 1.0,
                "source": source,
                "language": language,
            },
        ),
    )


def test_mix_span_examples_respects_shuffle_seed() -> None:
    examples_a = [
        _span_example("Alice 0912345678", (6, 16), entity_type="PHONE", source="a"),
        _span_example("Bob bob@example.com", (4, 19), entity_type="EMAIL", source="a"),
    ]
    examples_b = [
        _span_example("Carol A123456789", (6, 16), entity_type="ID", source="b"),
        _span_example("Dan 02-1234-5678", (4, 16), entity_type="PHONE", source="b"),
    ]
    sources = [("a", examples_a), ("b", examples_b)]

    combined = list(examples_a) + list(examples_b)

    # Guard against accidentally using global random state.
    state = random.getstate()
    try:
        random.seed(999)
        mixed = mix_span_examples(sources, shuffle=True, seed=123)
    finally:
        random.setstate(state)

    expected = list(combined)
    rng = random.Random(123)
    rng.shuffle(expected)

    assert mixed == expected


def test_mix_span_examples_can_disable_shuffle() -> None:
    examples_a = [_span_example("A A123", (2, 6), entity_type="ID", source="a")]
    examples_b = [_span_example("B B456", (2, 6), entity_type="ID", source="b")]
    sources = [("a", examples_a), ("b", examples_b)]

    mixed = mix_span_examples(sources, shuffle=False)
    assert mixed == list(examples_a) + list(examples_b)


def test_write_mixed_dataset_writes_expected_artifacts(tmp_path: Path) -> None:
    examples_a = [
        _span_example("Alice 0912345678", (6, 16), entity_type="PHONE", source="a"),
        _span_example("Bob bob@example.com", (4, 19), entity_type="EMAIL", source="a"),
    ]
    examples_b = [
        _span_example("Carol A123456789", (6, 16), entity_type="ID", source="b"),
    ]

    src_a_path = tmp_path / "src_a.jsonl"
    src_b_path = tmp_path / "src_b.jsonl"
    dump_span_examples_jsonl(src_a_path, examples_a)
    dump_span_examples_jsonl(src_b_path, examples_b)

    out_dir = tmp_path / "mixed"
    result = write_mixed_dataset(
        output_dir=out_dir,
        dataset_name="demo",
        split="train",
        sources=[
            SpanJsonlSource(name="dataset_a", jsonl_path=src_a_path, max_examples=1),
            SpanJsonlSource(name="dataset_b", jsonl_path=src_b_path, max_examples=0),
        ],
        shuffle=False,
        seed=0,
        max_issues=10,
    )

    jsonl_path = Path(result["jsonl_path"])
    manifest_path = out_dir / "manifest.json"
    quality_path = out_dir / "quality.json"

    assert jsonl_path.exists()
    assert manifest_path.exists()
    assert quality_path.exists()

    loaded = load_span_examples_jsonl(jsonl_path)
    assert [ex.text for ex in loaded] == [examples_a[0].text, examples_b[0].text]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"] == "demo"
    assert manifest["split"] == "train"
    assert manifest["count"] == 2
    assert manifest["sources"][0]["name"] == "dataset_a"
    assert manifest["sources"][0]["loaded"] == 1
    assert manifest["sources"][1]["name"] == "dataset_b"
    assert manifest["sources"][1]["loaded"] == 1

    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["summary"]["examples"] == 2
