from typing import TypedDict, Tuple, Literal, List

# 預定義 PII 類型常量
PII_TYPES = Literal['NAME', 'ID', 'PHONE', 'EMAIL', 'ADDRESS', 'UNIFIED_BUSINESS_NO']

class Entity(TypedDict):
    span: Tuple[int, int]   # [start, end]
    type: PII_TYPES        # 使用預定義類型
    score: float       # confidence'source: str
    source: str        # 添加來源標識 (如 "bert", "regex", "spacy")

class PIIDetector:
    def detect(self, text: str) -> List[Entity]:
        raise NotImplementedError
