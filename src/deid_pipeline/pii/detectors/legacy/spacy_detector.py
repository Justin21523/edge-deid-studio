from __future__ import annotations

import os
import re
from typing import List

from ....config import Config, load_regex_rules
from ....runtime.registry import get_spacy_pipeline
from ...utils import logger
from ...utils.base import Entity, PIIDetector

# Type mapping: spaCy label -> canonical PII type
SPACY_TO_PII_TYPE = {
    # spaCy native labels
    "PERSON":              "NAME",
    "GPE":                 "ADDRESS",
    "LOC":                 "ADDRESS",
    "ORG":                 "ORGANIZATION",
    # Custom labels injected via EntityRuler
    "PHONE":               "PHONE",
    "ID":                  "ID",
    "PASSPORT":            "PASSPORT",
    "UNIFIED_BUSINESS_NO": "UNIFIED_BUSINESS_NO",
    "EMAIL":               "EMAIL",
    "ADDRESS":             "ADDRESS",
    "MEDICAL_ID":          "MEDICAL_ID",
}

# Map YAML flag strings to re module flags.
FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE":  re.MULTILINE,
    "DOTALL":     re.DOTALL,
    # Add more flags here if needed.
}


class SpacyDetector(PIIDetector):
    def __init__(self, lang: str = "zh"):
        """Initialize spaCy pipeline (optional dependency).

        Args:
            lang: "zh" uses a Chinese pipeline (default: zh_core_web_sm),
                  "en" uses an English pipeline (default: en_core_web_sm).
        """

        self.lang = lang
        # Allow overriding model names via env vars.
        if lang == "zh":
            model_name = os.getenv("SPACY_ZH_MODEL", "zh_core_web_sm")
        else:
            model_name = os.getenv("SPACY_EN_MODEL", "en_core_web_sm")
        logger.info("Loading spaCy model '%s' for lang=%s", model_name, lang)
        self.nlp = get_spacy_pipeline(model_name)

        # Load regex rules and inject them into an EntityRuler.
        regex_file = (
            Config.REGEX_RULES_FILE if lang == "zh"
            else Config.REGEX_EN_RULES_FILE
        )
        raw_rules = load_regex_rules(regex_file)

        # Normalize: support list-of-dicts, list-of-strings, and dict forms.
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
                logger.warning("Unknown regex format for %s, skipping", ent)

        # Compile regex patterns for a separate backstop pass.
        self.regex_patterns = {}
        for ent_type, rules in norm_rules.items():
            compiled = []
            for rule in rules:
                pat = rule["pattern"]
                flags = FLAG_MAP.get(rule.get("flags"), 0)
                compiled.append(re.compile(pat, flags))
            self.regex_patterns[ent_type] = compiled

        # Inject patterns into spaCy via EntityRuler.
        rules_fingerprint = f"{model_name}:{str(regex_file)}"
        existing_fingerprint = getattr(self.nlp, "_edge_deid_rules_fingerprint", None)

        if existing_fingerprint != rules_fingerprint:
            if "entity_ruler" in getattr(self.nlp, "pipe_names", []):
                try:
                    self.nlp.remove_pipe("entity_ruler")
                except Exception:
                    pass

            try:
                if "ner" in getattr(self.nlp, "pipe_names", []):
                    ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                else:
                    ruler = self.nlp.add_pipe("entity_ruler")
            except Exception:
                ruler = self.nlp.get_pipe("entity_ruler")

            ruler_patterns = []
            for ent_type, rules in norm_rules.items():
                for rule in rules:
                    ruler_patterns.append(
                        {
                            "label": ent_type,
                            "pattern": [{"TEXT": {"REGEX": rule["pattern"]}}],
                        }
                    )
            try:
                ruler.add_patterns(ruler_patterns)
            except Exception as exc:
                logger.warning("Failed to add EntityRuler patterns: %s", exc)

            setattr(self.nlp, "_edge_deid_rules_fingerprint", rules_fingerprint)

    def detect(self, text: str) -> List[Entity]:
        ents: List[Entity] = []

        # 1) spaCy NER + EntityRuler detections
        doc = self.nlp(text)
        for e in doc.ents:
            if e.label_ in SPACY_TO_PII_TYPE:
                ents.append(Entity(
                    span=(e.start_char, e.end_char),
                    type=SPACY_TO_PII_TYPE[e.label_],
                    score=0.99,
                    source="spacy"
                ))

        # 2) Separate regex backstop (ensure high recall)
        for pii_type, patterns in self.regex_patterns.items():
            for pat in patterns:
                for m in pat.finditer(text):
                    ents.append(Entity(
                        span=(m.start(), m.end()),
                        type=pii_type,
                        score=1.0,
                        source="regex"
                    ))

        return sorted(ents, key=lambda e: e['span'][0])
