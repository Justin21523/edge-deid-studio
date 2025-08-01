# src/deid_pipeline/pii/detectors/composite.py
from typing import List
from ..utils.base import PIIDetector, Entity
from ...config import Config
from ..utils import logger

class CompositeDetector(PIIDetector):
    def __init__(self, *detectors: PIIDetector):
        self.detectors = detectors
        self.config = Config()

    def detect(self, text: str) -> List[Entity]:
        all_ents = []

        # 並行執行所有檢測器
        for detector in self.detectors:
            try:
                ents = detector.detect(text)
                all_ents.extend(ents)
                logger.debug(f"{detector.__class__.__name__} 找到 {len(ents)} 個實體")
            except Exception as e:
                logger.error(f"{detector.__class__.__name__} 檢測失敗: {str(e)}")

        # 解決實體衝突
        resolved_ents = self._resolve_conflicts(all_ents)
        return resolved_ents

    def _resolve_conflicts(self, entities: List[Entity]) -> List[Entity]:
        """
        用 ENTITY_PRIORITY 解決重疊：
        1) 先按起始位置排
        2) 如果兩個 span 有重疊，選 priority 較高的；一樣高就選 score 較高的
        """
        if not entities:
            return []

        # 按起始位置排序
        sorted_ents = sorted(entities, key=lambda e:  e["span"][0])
        resolved: List[Entity] = []
        for ent in sorted_ents:
            if not resolved:
                resolved.append(ent)
                continue

            last = resolved[-1]
            last_start, last_end = last['span']
            current_start, current_end = ent['span']

            overlap_ratio = 0
            # 檢查重疊
            if current_start < last_end:
                overlap = min(last_end, current_end) - current_start
                last_length = last_end - last_start
                current_length = current_end - current_start


                overlap_ratio = overlap / min(last_length, current_length)


                last_priority = self._get_priority(last['type'])
                current_priority = self._get_priority(ent['type'])

             # 決策邏輯
            if overlap_ratio > 0.5:
                if current_priority > last_priority:
                    resolved[-1] = ent
                elif current_priority == last_priority:
                    # same priority 就比較 score
                    if ent['score'] > last['score']:
                        resolved[-1] = ent
                    # score 一樣再比 source (短字串優先)
                    elif ent['score'] == last['score'] and len(ent['source']) < len(last['source']):
                        resolved[-1] = ent
            else:
                 # 部分重疊但未達到阈值，保留雙者
                resolved.append(ent)
        else:
             # 無重疊，直接加入
            resolved.append(ent)


        return resolved


    def _get_priority(self, entity_type: str) -> int:
         """獲取實體類型優先級"""
         return self.config.ENTITY_PRIORITY.get(entity_type, 50)

