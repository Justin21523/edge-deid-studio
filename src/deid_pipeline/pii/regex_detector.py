from typing import List
import yaml, re, os, logging
from .base import PIIDetector, Entity

class RegexDetector(PIIDetector):
    def __init__(self, config_path="configs/regex_zh.yaml"):
        self.config_path = config_path
        self.last_modified = 0
        self.logger = logging.getLogger(__name__)
        self.load_rules()

    def load_rules(self):
        """動態熱更新 YAML 規則檔，支援 flags 與來源標記"""
        mod_time = os.path.getmtime(self.config_path)
        if mod_time <= self.last_modified:
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)
        # 修改規則格式支援標誌
        # regex_zh.yaml 範例：
        # ID:
        #   - pattern: "[A-Z]\\d{9}"
        #     flags: IGNORECASE
        #   - pattern: "\\d{8}"
        #     flags: MULTILINE

        # 修改初始化：
        patterns = []
        for typ, rule_list in rules.items():
            for rule in rule_list:
                patt = rule["pattern"]
                flags = 0
                if "flags" in rule:
                    for fl in rule["flags"].split("|"):
                        flags |= getattr(re, fl.upper(), 0)
                patterns.append((typ, re.compile(patt, flags)))
        self.patterns = patterns
        self.last_modified = mod_time
        self.logger.info(f"[RegexDetector] 重新載入 {len(patterns)} 條規則")

    def detect(self, text: str) -> List[Entity]:
        # 每次執行前自動檢查是否有更新
        self.load_rules()
        ents=[]
        for typ, pat in self.patterns:
            for m in pat.finditer(text):
                ents.append({"span":[m.start(),m.end()],"type":typ,"score":1.0})
                ents.append(Entity(
                    span=(m.start(), m.end()),
                    type=typ,
                    score=1.0,
                    source="regex"
                ))
        return sorted(ents, key=lambda x: x["span"][0])
