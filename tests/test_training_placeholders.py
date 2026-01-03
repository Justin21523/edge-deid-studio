from __future__ import annotations

from deid_pipeline.training.placeholders import (
    canonicalize_placeholder_text,
    contains_cjk,
    replace_spans_with_placeholders,
)


def test_canonicalize_placeholder_text_normalizes_types() -> None:
    text = "Hi <LASTNAME_1>, email [EMAIL_ADDRESS]."
    assert canonicalize_placeholder_text(text) == "Hi <NAME>, email [EMAIL]."


def test_contains_cjk_detects_chinese_characters() -> None:
    assert contains_cjk("台北市信義路1號") is True
    assert contains_cjk("hello world") is False


def test_replace_spans_with_placeholders_replaces_from_end() -> None:
    text = "王小明 email bob@example.com."
    entities = [
        {"type": "NAME", "span": (0, 3), "text": "王小明"},
        {"type": "EMAIL_ADDRESS", "span": (10, 25), "text": "bob@example.com"},
    ]
    masked = replace_spans_with_placeholders(text, entities)
    assert masked == "<NAME> email <EMAIL>."

