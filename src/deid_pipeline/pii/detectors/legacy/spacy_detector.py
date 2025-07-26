# src/deid_pipeline/pii/detectors/legacy/spacy_detector.py
from typing import List
import spacy, re
from deid_pipeline.pii.utils.base import PIIDetector, Entity
from deid_pipeline.config import Config

# 新增統一類型映射
# type mapping: spaCy label → our PII_TYPES
SPACY_TO_PII_TYPE = {
    "PERSON": "NAME",
    "GPE": "ADDRESS",
    "ORG": "ORGANIZATION",
    "LOC": "ADDRESS"
}

# 預編譯我們的 regex 規則
PII_PATTERNS = {
    ent_type: [re.compile(p) for p in patterns]
    for ent_type, patterns in Config.REGEX_PATTERNS.items()
}

_nlp = spacy.load("en_core_web_sm")

class SpacyDetector(PIIDetector):
    def __init__(self):
        self._nlp = _nlp
        # 預編譯規則
        self.regex_patterns = PII_PATTERNS

    def detect(self, text: str) -> List[Entity]:
        ents: List[Entity] = []
        doc = self._nlp(text)
        # spaCy 偵測
        for e in doc.ents:
            if e.label_ in SPACY_TO_PII_TYPE:
                ents.append(Entity(
                    span=(e.start_char, e.end_char),
                    type=SPACY_TO_PII_TYPE[e.label_],
                    score=0.99,
                    source="spacy"
                ))
        # regex 偵測
        for typ, patterns in self.regex_patterns.items():
            for pat in patterns:
                for m in pat.finditer(text):
                    ents.append(Entity(
                        span=(m.start(), m.end()),
                        type=typ,
                        score=1.0,
                        source="regex"
                    ))
        # 去掉重疊，保留最高 score
        return self._resolve_conflicts(sorted(ents, key=lambda x: x["span"][0]))

    def _resolve_conflicts(self, entities: List[Entity]) -> List[Entity]:
        resolved: List[Entity] = []
        for ent in entities:
            if not resolved:
                resolved.append(ent); continue
            last = resolved[-1]
            # 若 overlap
            if ent["span"][0] < last["span"][1]:
                if ent["score"] > last["score"]:
                    resolved[-1] = ent
            else:
                resolved.append(ent)
        return resolved

def detect_pii(text: str) -> list[Entity]:
    return SpacyDetector().detect(text)
