import pytest
from deid_pipeline.pii.detectors import get_detector

@pytest.mark.parametrize("lang, text, expect", [
    ("en", "Alice lives in Taipei, email a@b.com", ["EMAIL"]),
    ("zh", "Phone 0912345678, ID A123456789", ["PHONE", "ID"]),
])
def test_detect(lang, text, expect):
    det = get_detector(lang)
    types = [e["type"] for e in det.detect(text)]
    for ex in expect:
        assert ex in types
