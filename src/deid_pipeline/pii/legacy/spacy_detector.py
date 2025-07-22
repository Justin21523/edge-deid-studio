# src/deid_pipeline/pii/legacy/spacy_detector.py
import spacy, re
from ..base import PIIDetector, Entity   # ← 你的抽象類別
from ... import USE_STUB

if USE_STUB:
    class _StubSpacy:
        def __call__(self, text): return []
    _nlp = _StubSpacy()
else:
    import spacy
    _nlp = spacy.load("en_core_web_sm")


# 可加入正則規則（電話、email、身分證、信用卡等）
_PII_PATTERNS = {
    "PHONE": [
        r"\b09\d{8}\b",
        r"\b\d{3}-\d{3}-\d{4}\b",
        r"\+9\d{8}\b"
    ],
    "EMAIL": [
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    ],
    "ID": [
        r"\b[A-Z][1-2]\d{8}\b"
    ]
}


class SpacyDetector(PIIDetector):
    def detect(self, text: str) -> list[Entity]:
        doc = _nlp(text)
        ents = [
            {"span": [e.start_char, e.end_char], "type": e.label_, "score": 0.99}
            for e in doc.ents if e.label_ in ["PERSON", "GPE", "ORG", "LOC"]
        ]
        for typ, patterns in _PII_PATTERNS.items():         # regex 部分
            for pat in patterns:
                for m in re.finditer(pat, text):
                    ents.append({"span": [m.start(), m.end()], "type": typ, "score": 1.0})
        return sorted(ents, key=lambda x: x["span"][0])


def detect_pii(text: str) -> list[Entity]:
    return SpacyDetector().detect(text)
