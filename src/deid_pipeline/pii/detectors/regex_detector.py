# src/deid_pipeline/pii/detectors/regex_detector.py
import os
import yaml
import re
from pathlib import Path
from typing import List
from ..utils.base import PIIDetector, Entity
from ...config import Config
from ..utils import logger

class RegexDetector(PIIDetector):
    def __init__(self, config_path: str = None):
        self.config = Config()
        # 預設使用 config 裡設定的路徑
        self.config_path = Path(config_path) if config_path else self.config.REGEX_RULES_FILE
        self.last_modified = 0
        self.patterns = []
        self.load_rules()

    def load_rules(self):
        """載入正則規則，支援熱更新"""
        try:
            mod_time = os.path.getmtime(self.config_path)
            if mod_time <= self.last_modified:
                return

            # 支援 list-of-dicts 與單純 dict[str→list of str]
            with open(self.config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            rules = {}
            for typ, body in raw.items():
                if isinstance(body, list) and all(isinstance(i, dict) for i in body):
                    # already list of {"pattern":..., "flags":...}
                    rules[typ] = body
                elif isinstance(body, list) and all(isinstance(i, str) for i in body):
                    # shorthand list of str → wrap as dict
                    rules[typ] = [{"pattern": b} for b in body]
                elif isinstance(body, str):
                    # single pattern str → wrap into list
                    rules[typ] = [{"pattern": body}]
                else:
                    logger.warning(f"Unknown regex format for {typ}, skipping")
                    continue

            self.patterns = []
            for typ, rule_list in rules.items():
                for rule in rule_list:
                    pattern = rule["pattern"]
                    flags = 0

                    # 處理正則標誌
                    if "flags" in rule:
                        for flag in rule["flags"].split("|"):
                            flag = flag.strip().upper()
                            if hasattr(re, flag):
                                flags |= getattr(re, flag)

                    try:
                        compiled = re.compile(pattern, flags)
                        self.patterns.append((typ, compiled))
                        logger.debug(f"編譯正則成功: {typ} - {pattern}")
                    except Exception as e:
                        logger.error(f"正則編譯失敗: {pattern}, 錯誤: {str(e)}")

            self.last_modified = mod_time
            logger.info(f"已載入 {len(self.patterns)} 條正則規則")

        except Exception as e:
            logger.error(f"載入正則規則失敗: {str(e)}")
            self.patterns = []

    def detect(self, text: str) -> List[Entity]:
        self.load_rules()  # 檢查是否需要重新載入規則
        entities: List[Entity] = []

        for typ, pattern in self.patterns:
            for match in pattern.finditer(text):
                entities.append(Entity(
                    span=(match.start(), match.end()),
                    type=typ,
                    score=1.0,
                    source="regex",
                ))

        return entities
