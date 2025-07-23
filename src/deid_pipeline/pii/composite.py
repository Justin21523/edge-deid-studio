from typing import List
from .base import PIIDetector, Entity
import asyncio

class CompositeDetector(PIIDetector):
    def __init__(self, *detectors: PIIDetector):
        self.detectors = detectors

    def detect(self, text: str) -> List[Entity]:
        # 同步版：收集所有實體
        all_ents = [e for d in self.detectors for e in d.detect(text)]
        # 依起始位置排序，方便重疊檢查
        sorted_ents = sorted(all_ents, key=lambda e: e["span"][0])
        resolved = []
        for ent in sorted_ents:
            if not resolved:
                resolved.append(ent)
                continue
            last = resolved[-1]
            # 檢查是否有 span 重疊
            if ent["span"][0] < last["span"][1]:
                # 若重疊，優先級高者取代；若同級則取較高 score
                p_ent  = self.get_entity_priority(ent)
                p_last = self.get_entity_priority(last)
                if p_ent > p_last or (p_ent == p_last and ent["score"] > last["score"]):
                    resolved[-1] = ent
                # 否則保留原有實體
            else:
                resolved.append(ent)
        return resolved

    def get_entity_priority(self, entity: Entity) -> int:
        """定義實體類型優先級，數值越大優先級越高"""
        priority_map = {
            "ID":100,
            "PHONE":90,
            "NAME":80,
            "ADDRESS":70,
            "EMAIL":60
        }
        return priority_map.get(entity["type"], 50)

    async def adetect(self, text: str) -> list[Entity]:
        """非同步版本，可並行觸發多個 detector"""
        tasks = [d.adetect(text) for d in self.detectors if hasattr(d, "adetect")]
        results = await asyncio.gather(*tasks)
        # 平坦化後套用相同邏輯
        flat = [e for sub in results for e in sub]
        return self.detect(text)  # 直接呼叫同步版合併
