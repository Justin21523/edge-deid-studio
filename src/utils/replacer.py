from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from .fake_provider import GPT2Provider

class Replacer:
    """Deprecated legacy replacer.

    Prefer using `deid_pipeline.pii.utils.replacer.Replacer` for new code.
    """

    def __init__(self, provider: Optional[GPT2Provider] = None):
        self.provider = provider or GPT2Provider()
        self.cache: Dict[str, str] = {}

    def replace(
        self, text: str, entities: List[Dict], mode: str = "replace"
    ) -> Tuple[str, List[Dict]]:
        offset = 0
        events: List[Dict] = []
        for ent in sorted(entities, key=lambda e: e['span'][0]):
            s, e = ent['span']
            fake = self.get_fake(ent['type'], text[s:e])
            if mode == 'replace':
                text = text[:s+offset] + fake + text[e+offset:]
                events.append({'old':(s+offset, e+offset), 'new':fake})
                offset += len(fake) - (e - s)
            else:   # black
                events.append({'span':(s+offset, e+offset)})
        return text, events

    def get_fake(self, entity_type: str, original: str) -> str:
        key = f"{entity_type}:{original}"
        if key not in self.cache:
            self.cache[key] = self.provider.generate(entity_type, original)
        return self.cache[key]

    @staticmethod
    def dumps(events):
        return json.dumps(events, ensure_ascii=False, indent=2)
