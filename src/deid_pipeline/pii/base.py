from typing import TypedDict, List

class Entity(TypedDict):
    span: List[int]    # [start, end]
    type: str          # PII 類型
    score: float       # confidence

class PIIDetector:
    def detect(self, text: str) -> List[Entity]:
        raise NotImplementedError
