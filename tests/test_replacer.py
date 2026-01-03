from deid_pipeline.pii.detectors import get_detector
from deid_pipeline.pii.utils.replacer import Replacer

def test_replace_consistency():
    text = "ID A123456789 and again A123456789. Phone 0912345678."
    entities = get_detector("zh").detect(text)

    new_text, events = Replacer().replace(text, entities, mode="replace")
    assert "A123456789" not in new_text

    id_replacements = [
        e["replacement"] for e in events if e.get("original") == "A123456789"
    ]
    assert len(id_replacements) >= 2
    assert len(set(id_replacements)) == 1

def test_blackbox_mode_keeps_length():
    text = "ID A123456789 and Phone 0912345678."
    entities = get_detector("zh").detect(text)

    masked, _events = Replacer().replace(text, entities, mode="blackbox")
    assert len(masked) == len(text)
    assert "A123456789" not in masked
    assert "0912345678" not in masked
