from typing import List, Dict
from .fake_provider import FakerProvider, GPT2Provider
import json

class Replacer:
    def __init__(self, provider=None):
        self.provider = provider or GPT2Provider()
        self.cache = {}

    def replace(self, text:str, entities:List[Dict], mode='replace'):
        offset = 0
        events = []
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
