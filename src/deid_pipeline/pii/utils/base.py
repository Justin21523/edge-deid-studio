# src/deid_pipeline/pii/utils/base.py
from typing import TypedDict, Tuple, Literal, List

# 預定義 PII 類型常量
PII_TYPES = Literal['NAME', 'ID', 'PHONE', 'EMAIL', 'ADDRESS',
                    'UNIFIED_BUSINESS_NO','PASSPORT','MEDICAL_ID','CONTRACT_NO']
# 新增 PII 類型：
# - PASSPORT: 護照號
# - MEDICAL_ID: 病歷號
# - CONTRACT_NO: 合約編號
class Entity(TypedDict):
    span: Tuple[int, int]  # [start, end]
    type: PII_TYPES        # 使用預定義類型
    score: float           # confidence
    source: str            # 添加來源標識 (如 "bert", "regex", "spacy")

class PIIDetector:
    def detect(self, text: str) -> List[Entity]:
        raise NotImplementedError
