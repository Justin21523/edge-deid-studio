from __future__ import annotations

from deid_pipeline.training.datasets import SpanExample
from deid_pipeline.training.quality import analyze_span_examples, validate_span_example


def test_validate_span_example_flags_common_issues():
    text = "abcdef"
    entities = [
        {"type": "NAME", "span": (0, 2), "text": "ab"},
        {"type": "", "span": (2, 4), "text": "cd"},
        {"type": "ID", "span": (4, 4), "text": ""},
        {"type": "PHONE", "span": (4, 9), "text": "ef"},
        {"type": "EMAIL", "span": (1, 3), "text": "xx"},
    ]

    issues = validate_span_example(text, entities)
    kinds = {i.get("kind") for i in issues}

    assert "missing_type" in kinds
    assert "empty_span" in kinds
    assert "span_out_of_bounds" in kinds
    assert "text_mismatch" in kinds


def test_analyze_span_examples_summarizes_counts():
    examples = [
        SpanExample(text="abc", entities=tuple([{"type": "NAME", "span": (0, 2), "text": "ab"}])),
        SpanExample(
            text="abcdef",
            entities=tuple(
                [
                    {"type": "ID", "span": (0, 4), "text": "abcd"},
                    {"type": "PHONE", "span": (2, 5), "text": "cde"},
                ]
            ),
        ),
        SpanExample(text="xyz", entities=tuple([{"type": "EMAIL", "span": (0, 10), "text": "xyz"}])),
    ]

    report = analyze_span_examples(examples, max_issues=50)
    summary = report["summary"]

    assert summary["examples"] == 3
    assert summary["entities"] == 4

    issue_counts = summary["issue_counts"]
    assert issue_counts["overlap"] >= 1
    assert issue_counts["span_out_of_bounds"] >= 1

    entity_types = summary["entity_type_counts"]
    assert entity_types["NAME"] == 1
    assert entity_types["ID"] == 1
    assert entity_types["PHONE"] == 1
    assert entity_types["EMAIL"] == 1

