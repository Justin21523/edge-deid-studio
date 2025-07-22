import re, yaml, pathlib
from .base import PIIDetector, Entity

class RegexDetector(PIIDetector):
    def __init__(self, cfg: str | pathlib.Path = "configs/regex_zh.yaml"):
        patterns = yaml.safe_load(open(cfg, encoding="utf-8"))
        self.regexes = {t: re.compile(p) for t, p in patterns.items()}

    def detect(self, text: str) -> list[Entity]:
        out = []
        for typ, rgx in self.regexes.items():
            for m in rgx.finditer(text):
                out.append({"span": [m.start(), m.end()], "type": typ, "score": 1.0})
        return out
