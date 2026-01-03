from __future__ import annotations

from deid_pipeline import DeidPipeline
from deid_pipeline.core.contracts import normalize_entity, replacement_key


def test_normalize_entity_adds_confidence_and_language():
    raw = {"type": "ID", "span": [0, 10], "score": 1.0, "source": "regex"}
    normalized = normalize_entity(raw, language="zh", text="A1234567890")

    assert normalized["type"] == "ID"
    assert normalized["span"] == (0, 10)
    assert normalized["confidence"] == 1.0
    assert normalized["score"] == 1.0
    assert normalized["language"] == "zh"
    assert normalized["text"] == "A123456789"


def test_pipeline_returns_deid_result_contract(tmp_path):
    input_path = tmp_path / "sample.txt"
    input_path.write_text("ID A123456789 and phone 0912345678.", encoding="utf-8")

    result = DeidPipeline(language="zh").process(str(input_path), output_mode="replace")

    assert isinstance(result.text, str)
    assert isinstance(result.entities, list)
    assert isinstance(result.replacement_map, dict)
    assert isinstance(result.events, list)
    assert isinstance(result.timings_ms, dict)
    assert isinstance(result.artifacts, dict)
    assert result.schema_version == "1.0"

    # Entities should be normalized (language + confidence present).
    assert any(e["type"] == "ID" for e in result.entities)
    for entity in result.entities:
        assert entity["language"] == "zh"
        assert "confidence" in entity
        assert "score" in entity

    # Replacement map uses phase-1 keys.
    id_key = replacement_key("ID", "A123456789")
    assert id_key in result.replacement_map
    assert result.replacement_map[id_key] != "A123456789"

    # Timings should include the main stages.
    for key in ["extract", "detect", "replace", "total"]:
        assert key in result.timings_ms
        assert result.timings_ms[key] >= 0.0

