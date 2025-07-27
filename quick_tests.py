# quick_tests.py
from pathlib import Path

from src.deid_pipeline.parser.text_extractor import extract_text
from src.deid_pipeline.pii.detectors import get_detector
from src.deid_pipeline.pii.utils.replacer import Replacer

def test_detector():
    cases = [
        ("zh", "王小明身分證A123456789，手機0912345678", {"ID", "PHONE"}),
        ("en", "Alice lives in Taipei, email alice@mail.com", {"EMAIL", "ADDRESS"}),
    ]
    for lang, txt, expected in cases:
        det = get_detector(lang)
        found = {e["type"] for e in det.detect(txt)}
        assert expected <= found, f"{lang} det failed: expect {expected}, got {found}"
    print("✅ Detector tests passed")

def test_replacer():
    txt = "王小明的電話0912345678"
    det = get_detector("zh").detect(txt)
    new_txt, _ = Replacer().replace(txt, det, mode="replace")
    assert "0912345678" not in new_txt
    print("✅ Replacer tests passed")

def test_extractor():
    sample = Path("test_input/112–實習–extracted.txt")
    text, _ = extract_text(sample)
    assert len(text) > 20
    print("✅ Text extractor tests passed")

if __name__ == "__main__":
    test_detector()
    test_replacer()
    test_extractor()
    print("🎉 All quick tests OK")
