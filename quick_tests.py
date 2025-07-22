"""
快速自我檢查：
1. Detector zh/en 抓 PII
2. Replacer 一致性
3. Text extractor 對 PDF/TXT
執行：python quick_tests.py
"""
from pathlib import Path
from deid_pipeline.pii import get_detector
from deid_pipeline.utils.replacer import Replacer
from deid_pipeline.parser.text_extractor import extract_text

def test_detector():
    cases = [
        ("en", "Alice lives in Taipei, email alice@mail.com", {"PERSON","EMAIL"}),
        ("zh", "王小明身分證A123456789，手機0912345678", {"NAME","ID","PHONE"}),
    ]
    for lang, txt, expect in cases:
        det = get_detector(lang)
        found = {e["type"] for e in det.detect(txt)}
        assert expect <= found, f"{lang} 應有 {expect}, 只找到 {found}"
    print("✅ Detector 測試通過")

def test_replacer():
    txt = "王小明借給王小明 0912345678"
    det = get_detector("zh").detect(txt)
    new_txt, _ = Replacer().replace(txt, det)
    assert "王小明" not in new_txt and "0912345678" not in new_txt
    names = [w for w in new_txt.split() if len(w) >= 3]
    assert len(set(names)) == 1          # 同名一致
    print("✅ Replacer 測試通過")

def test_extractor():
    sample = Path("test_input/112-1_實習一_extracted.txt")
    txt = extract_text(sample)
    assert len(txt) > 20
    print("✅ Text Extractor 測試通過")


if __name__ == "__main__":
    test_detector()
    test_replacer()
    test_extractor()
    print("🎉 全部 quick test OK")
