from __future__ import annotations

import pytest

from deid_pipeline.training.datasets import HuggingFaceTokenNERAdapter, TokenNERExample, token_examples_to_span_examples
from deid_pipeline.training.masked_pairs import extract_entities_from_masked_pair


def test_masked_pair_extraction_finds_spans_and_types():
    original = "My name is John Smith and phone 0912345678."
    masked = "My name is <NAME> and phone <PHONE>."

    entities = extract_entities_from_masked_pair(original, masked, language="en")
    types = {e["type"] for e in entities}

    assert "NAME" in types
    assert "PHONE" in types

    for ent in entities:
        start, end = ent["span"]
        assert original[start:end] == ent["text"]


def test_token_adapter_is_network_gated_by_default():
    adapter = HuggingFaceTokenNERAdapter("tner/wikiann", config_name="en")
    with pytest.raises(RuntimeError):
        adapter.load(split="train", allow_network=False)


def test_token_examples_to_span_examples_builds_entities():
    token_examples = [
        TokenNERExample(tokens=("John", "Smith", "works"), tags=("B-NAME", "I-NAME", "O"))
    ]
    span_examples = token_examples_to_span_examples(token_examples, language="en")
    assert len(span_examples) == 1
    ex = span_examples[0]
    assert ex.text == "John Smith works"
    assert len(ex.entities) == 1
    ent = ex.entities[0]
    assert ent["type"] == "NAME"
    assert ex.text[ent["span"][0] : ent["span"][1]] == "John Smith"
