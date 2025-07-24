import pytest
from deid_pipeline.pii.detectors import get_detector

@pytest.mark.parametrize("lang, text, expect", [
    ("en", "Alice 住在 Taipei, email a@b.com", ["PERSON","GPE","EMAIL"]),
    ("zh", "王小明手機0912345678, 身分證A123456789", ["PHONE","ID"]),
])
def test_detect(lang, text, expect):
    det = get_detector(lang)
    types = [e["type"] for e in det.detect(text)]
    for ex in expect:
        assert ex in types
