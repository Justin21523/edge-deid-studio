from __future__ import annotations

from deid_pipeline.pii.utils.replacer import Replacer


def test_replacement_is_deterministic_for_same_context_hash():
    text = "ID A123456789 and again A123456789."
    entities = [{"span": [3, 13], "type": "ID", "score": 1.0, "source": "regex"}]

    out1, events1 = Replacer().replace(text, entities, context_hash="doc-1")
    out2, events2 = Replacer().replace(text, entities, context_hash="doc-1")

    assert out1 == out2
    assert events1[0]["replacement"] == events2[0]["replacement"]


def test_replacement_changes_across_context_hashes():
    text = "ID A123456789."
    entities = [{"span": [3, 13], "type": "ID", "score": 1.0, "source": "regex"}]

    _out1, events1 = Replacer().replace(text, entities, context_hash="doc-A")
    _out2, events2 = Replacer().replace(text, entities, context_hash="doc-B")

    assert events1[0]["replacement"] != events2[0]["replacement"]

