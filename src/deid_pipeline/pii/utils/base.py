# src/deid_pipeline/pii/utils/base.py
from dataclasses import dataclass, field
from typing import TypedDict, Tuple, Dict, Literal, List, Optional, Any

# 預定義 PII 類型常量
PII_TYPES = Literal[
  "NAME","ID","PHONE","EMAIL","ADDRESS",
  "UNIFIED_BUSINESS_NO","TW_ID",
  "PASSPORT","MEDICAL_ID","CONTRACT_NO","ORGANIZATION"
]

# 新增 PII 類型：
# - PASSPORT: 護照號
# - MEDICAL_ID: 病歷號
# - CONTRACT_NO: 合約編號
# - ORGANIZATION: 機構公司編號

@dataclass
class Entity:
    """
    通用 PII 偵測結果物件
    """
    text: str                            # 偵測到的原始文字
    weight: int
    entity_type: str                     # 類型標籤 (e.g. "EMAIL_ADDRESS", "PHONE_NUMBER")
    start: Optional[int] = None          # 在整段文字中的起始 index (字元位置)
    end: Optional[int] = None            # 在整段文字中的結束 index
    confidence: float = 1.0              # 偵測信心度 (0~1)
    sources: Optional[str] = None        # 負責偵測此實體的 detector 名稱
    weight: float = 1.0                  # 該 detector 的權重，用於 composite 決策
    context: Optional[str] = None        # 前後文摘取 (方便後續審閱)
    bbox: Optional[Tuple[float,float,float,float]] = None
    # 若來自 OCR，可填歸一化後的 (x0, y0, x1, y1)
    page_num: Optional[int] = None       # 若處理多頁文件，可紀錄頁碼
    file_path: Optional[str] = None      # 來源檔案路徑
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 其他彈性欄位，如替換後文字(replacement)、entity_id 等

    def to_dict(self) -> Dict[str, Any]:
        """方便轉成 JSON / dict 輸出或序列化"""
        return {
            "text":            self.text,
            "entity_type":     self.entity_type,
            "start":           self.start,
            "end":             self.end,
            "confidence":      self.confidence,
            "context":         self.context,
            "bbox":            self.bbox,
            "page_num":        self.page_num,
            "file_path":       self.file_path,
            "metadata":        self.metadata,
        }

class PIIDetector:
    def detect(self, text: str) -> List[Entity]:
        raise NotImplementedError
