import os
import yaml
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List
from deid_pipeline.pii.utils.base import PIIDetector, Entity
from deid_pipeline.config import Config

logger = logging.getLogger(__name__)

class RegexDetector(PIIDetector):
    def __init__(self, config_path: str = None):
        self.config = Config()
        self.config_path = config_path or "configs/regex_zh.yaml"
        self.last_modified = 0
        self.patterns = []
        self.load_rules()

    def load_rules(self):
        """載入正則規則，支援熱更新"""
        try:
            mod_time = os.path.getmtime(self.config_path)
            if mod_time <= self.last_modified:
                return

            with open(self.config_path, "r", encoding="utf-8") as f:
                rules = yaml.safe_load(f)

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
        entities = []

        for typ, pattern in self.patterns:
            for match in pattern.finditer(text):
                entities.append({
                    "span": [match.start(), match.end()],
                    "type": typ,
                    "score": 1.0,
                    "source": "regex"
                })

        return entities
