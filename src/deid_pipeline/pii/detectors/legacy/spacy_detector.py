# src/deid_pipeline/pii/detectors/legacy/spacy_detector.py
from typing import List
import spacy
import os, re
from ...utils.base import PIIDetector, Entity
from ....config import Config, load_regex_rules
from ...utils import logger

# 新增統一類型映射
# type mapping: spaCy label → our PII_TYPES
SPACY_TO_PII_TYPE = {
    # spaCy 原生
    "PERSON":              "NAME",
    "GPE":                 "ADDRESS",
    "LOC":                 "ADDRESS",
    "ORG":                 "ORGANIZATION",
    # 以下是我們要注入的自訂 labels
    "PHONE":               "PHONE",
    "ID":                  "ID",
    "PASSPORT":            "PASSPORT",
    "UNIFIED_BUSINESS_NO": "UNIFIED_BUSINESS_NO",
    "EMAIL":               "EMAIL",
    "ADDRESS":             "ADDRESS",
    "MEDICAL_ID":          "MEDICAL_ID",
}

# 將 YAML 裡的 flags 字串對應到 re.FLAGS
FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE":  re.MULTILINE,
    "DOTALL":     re.DOTALL,
    # 如果還有其他 flag，可以補在這裡
}

class SpacyDetector(PIIDetector):
    def __init__(self, lang: str = "zh"):
        """
        lang: 'zh' 用中文模型 (zh_core_web_sm)
              'en' 用英文模型 (en_core_web_sm)
        """
        self.lang = lang
        # 允許透過 Config 覆寫成任意 spaCy model name
        if lang == "zh":
            model_name = os.getenv("SPACY_ZH_MODEL", "zh_core_web_sm")
        else:
            model_name = os.getenv("SPACY_EN_MODEL", "en_core_web_sm")
        logger.info(f"Loading spaCy model '{model_name}' for lang={lang}")
        self.nlp = spacy.load(model_name)

        # 讀取正則規則檔 (List[Dict])
        regex_file = (
            Config.REGEX_RULES_FILE if lang == "zh"
            else Config.REGEX_EN_RULES_FILE
        )
        raw_rules = load_regex_rules(regex_file)

        # ── Normalize: 支援 List[Dict], List[str], 以及單一 Dict[str,Any]
        norm_rules = {}
        for ent, body in raw_rules.items():
            # already list of dict: [{"pattern":..., "flags":...}, ...]
            if isinstance(body, list) and all(isinstance(i, dict) for i in body):
                norm_rules[ent] = body
            # list of strings: ["pat1","pat2"]
            elif isinstance(body, list) and all(isinstance(i, str) for i in body):
                norm_rules[ent] = [{"pattern": pat} for pat in body]
            # single dict: {"pattern":..., "flags":...}
            elif isinstance(body, dict):
                norm_rules[ent] = [body]
            else:
                logger.warning(f"Unknown regex format for {ent}, skipping")

        # 4) 用 normalize 後的規則編譯
        self.regex_patterns = {}
        for ent_type, rules in norm_rules.items():
            compiled = []
            for rule in rules:
                pat = rule["pattern"]
                flags = FLAG_MAP.get(rule.get("flags"), 0)
                compiled.append(re.compile(pat, flags))
            self.regex_patterns[ent_type] = compiled

        # 用 EntityRuler 把這些 regex patterns 注入到 spaCy pipeline
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler_patterns = []
        for ent_type, rules in norm_rules.items():
            for rule in rules:
                ruler_patterns.append({
                    "label": ent_type,
                    "pattern": [{"TEXT": {"REGEX": rule["pattern"]}}]
                })
        ruler.add_patterns(ruler_patterns)

    def detect(self, text: str) -> List[Entity]:
        ents: List[Entity] = []

        # 1) spaCy NER + EntityRuler 偵測
        doc = self.nlp(text)
        for e in doc.ents:
            if e.label_ in SPACY_TO_PII_TYPE:
                ents.append(Entity(
                    span=(e.start_char, e.end_char),
                    type=SPACY_TO_PII_TYPE[e.label_],
                    score=0.99,
                    source="spacy"
                ))

        # 2) 獨立的 regex 偵測 (確保沒漏)
        for pii_type, patterns in self.regex_patterns.items():
            for pat in patterns:
                for m in pat.finditer(text):
                    ents.append(Entity(
                        span=(m.start(), m.end()),
                        type=pii_type,
                        score=1.0,
                        source="regex"
                    ))
        # 去掉重疊，保留最高 score
        return sorted(ents, key=lambda e: e['span'][0])
