import yaml, re
from .base import PIIDetector, Entity

class RegexDetector(PIIDetector):
    def __init__(self, config_path="configs/regex_zh.yaml"):
        rules = yaml.safe_load(open(config_path))
        self.patterns = [(typ, re.compile(pat)) for typ, pats in rules.items() for pat in pats]

    def detect(self, text: str) -> List[Entity]:
        ents=[]
        for typ, pat in self.patterns:
            for m in pat.finditer(text):
                ents.append({"span":[m.start(),m.end()],"type":typ,"score":1.0})
        return sorted(ents, key=lambda x: x["span"][0])
