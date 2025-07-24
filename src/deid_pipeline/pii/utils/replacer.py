from .base import Entity
from .fake_provider import FakeProvider
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class Replacer:
    def __init__(self, provider=None):
        self.provider = provider or FakeProvider()
        self.cache: Dict[str, str] = {}

    def replace(self, text: str, entities: List[Entity], mode: str = 'replace') -> Tuple[str, List[dict]]:
        """
        替換文字中的PII實體

        參數:
            text: 原始文字
            entities: 檢測到的PII實體
            mode: 'replace' 或 'blackbox'

        返回:
            Tuple[替換後文字, 替換事件列表]
        """
        if mode == 'blackbox':
            return self._blackbox_mode(text, entities)
        return self._replace_mode(text, entities)

    def _replace_mode(self, text: str, entities: List[Entity]) -> Tuple[str, List[dict]]:
        """完整替換模式"""
        # 按位置反向排序以避免偏移問題
        sorted_ents = sorted(entities, key=lambda x: x["span"][0], reverse=True)
        new_text = text
        events = []

        for ent in sorted_ents:
            start, end = ent["span"]
            original = text[start:end]

            # 取得或生成假資料
            fake = self.provider.generate(ent["type"], original)

            # 執行替換
            new_text = new_text[:start] + fake + new_text[end:]

            # 記錄替換事件
            events.append({
                "original": original,
                "fake": fake,
                "type": ent["type"],
                "span": (start, start + len(fake)),  # 新位置
                "source": ent.get("source", "unknown")
            })

        return new_text, events

    def _blackbox_mode(self, text: str, entities: List[Entity]) -> Tuple[str, List[dict]]:
        """黑框模式（遮蔽模式）"""
        # 按位置反向排序以避免偏移問題
        sorted_ents = sorted(entities, key=lambda x: x["span"][0], reverse=True)
        new_text = text
        events = []

        for ent in sorted_ents:
            start, end = ent["span"]

            # 創建黑框（使用等長█字元）
            blackbox = "█" * (end - start)
            new_text = new_text[:start] + blackbox + new_text[end:]

            # 記錄事件
            events.append({
                "type": ent["type"],
                "span": (start, start + len(blackbox)),
                "source": ent.get("source", "unknown")
            })

        return new_text, events
