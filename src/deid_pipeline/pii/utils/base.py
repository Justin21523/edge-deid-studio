from __future__ import annotations

from typing import List, Literal, Tuple, TypedDict


PII_TYPES = Literal[
    "NAME",
    "ID",
    "PHONE",
    "EMAIL",
    "ADDRESS",
    "UNIFIED_BUSINESS_NO",
    "TW_ID",
    "PASSPORT",
    "MEDICAL_ID",
    "CONTRACT_NO",
    "ORGANIZATION",
]


class Entity(TypedDict):
    """Detector-level entity contract (minimal fields)."""

    span: Tuple[int, int]  # (start, end) in extracted text coordinates
    type: PII_TYPES
    score: float  # confidence score
    source: str  # e.g. "bert", "onnx", "regex", "spacy"


class PIIDetector:
    def detect(self, text: str) -> List[Entity]:
        raise NotImplementedError
