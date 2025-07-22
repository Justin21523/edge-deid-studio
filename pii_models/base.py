from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

Entity = Dict[str, object]  # {'span': (int,int), 'type': str, 'score': float}

class PIIDetector(ABC):
    """所有 PII 偵測器都繼承它，方便熱插拔。"""

    @abstractmethod
    def detect(self, text: str) -> List[Entity]:
        """回傳實體清單。"""
        raise NotImplementedError
