from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from ...config import Config
from ..utils import logger
from ..utils.base import Entity, PIIDetector


class RegexDetector(PIIDetector):
    def __init__(self, config_path: str | Path | None = None):
        self.config = Config()
        # Default to the path configured in Config.
        self.config_path = Path(config_path) if config_path else self.config.REGEX_RULES_FILE
        self.last_modified = 0
        self.patterns: list[tuple[str, re.Pattern[str]]] = []
        self.load_rules()

    def load_rules(self):
        """Load regex rules and support hot reload when the YAML changes."""

        try:
            mod_time = os.path.getmtime(self.config_path)
            if mod_time <= self.last_modified:
                return

            # Support list-of-dicts and shorthand list-of-strings forms.
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
                    logger.warning("Unknown regex format for %s, skipping", typ)
                    continue

            self.patterns = []
            for typ, rule_list in rules.items():
                for rule in rule_list:
                    pattern = rule["pattern"]
                    flags = 0

                    # Resolve regex flags (e.g. "IGNORECASE|MULTILINE").
                    if "flags" in rule:
                        for flag in rule["flags"].split("|"):
                            flag = flag.strip().upper()
                            if hasattr(re, flag):
                                flags |= getattr(re, flag)

                    try:
                        compiled = re.compile(pattern, flags)
                        self.patterns.append((typ, compiled))
                        logger.debug("Compiled regex: %s - %s", typ, pattern)
                    except Exception as exc:
                        logger.error("Failed to compile regex: %s (%s)", pattern, exc)

            self.last_modified = mod_time
            logger.info("Loaded %d regex rules", len(self.patterns))

        except Exception as exc:
            logger.error("Failed to load regex rules: %s", exc)
            self.patterns = []

    def detect(self, text: str) -> list[Entity]:
        self.load_rules()  # check for updated YAML rules
        entities: list[Entity] = []

        for typ, pattern in self.patterns:
            for match in pattern.finditer(text):
                entities.append(Entity(
                    span=(match.start(), match.end()),
                    type=typ,
                    score=1.0,
                    source="regex",
                ))

        return entities
