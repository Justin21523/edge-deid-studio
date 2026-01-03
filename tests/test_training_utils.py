from __future__ import annotations

from deid_pipeline.training.synthetic import generate_synthetic_span_examples
from deid_pipeline.training.tokenization import align_entities_to_tokens, build_bio_label_list


def test_generate_synthetic_span_examples_is_deterministic():
    a = generate_synthetic_span_examples(num_examples=3, seed=123, language="zh")
    b = generate_synthetic_span_examples(num_examples=3, seed=123, language="zh")
    assert a == b


def test_generate_synthetic_span_examples_spans_match_text():
    examples = generate_synthetic_span_examples(num_examples=5, seed=0, language="en")
    for ex in examples:
        text = ex["text"]
        for ent in ex["entities"]:
            start, end = ent["span"]
            assert text[start:end] == ent["text"]


def test_align_entities_to_tokens_builds_labels():
    labels = build_bio_label_list(["NAME"])
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    entities = [{"type": "NAME", "span": (0, 4), "text": "John"}]
    offset_mapping = [(0, 4), (5, 8), (0, 0)]
    label_ids = align_entities_to_tokens(
        entities=entities,
        offset_mapping=offset_mapping,
        label_to_id=label_to_id,
    )

    assert label_ids[0] == label_to_id["B-NAME"]
    assert label_ids[1] == label_to_id["O"]
    assert label_ids[2] == -100
