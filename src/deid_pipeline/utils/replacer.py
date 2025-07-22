from typing import List, Dict
from .fake_provider import FakerProvider

class Replacer:
    def __init__(self, provider=None):
        self.provider = provider or FakerProvider()

    def replace(self, text:str, entities:List[Dict], mode='replace'):
        offset = 0
        events = []
        for ent in sorted(entities, key=lambda e: e['span'][0]):
            s, e = ent['span']
            fake = self.provider.fake(ent['type'], text[s:e])
            if mode == 'replace':
                text = text[:s+offset] + fake + text[e+offset:]
                events.append({'old':(s+offset, e+offset), 'new':fake})
                offset += len(fake) - (e - s)
            else:   # black
                events.append({'span':(s+offset, e+offset)})
        return text, events
