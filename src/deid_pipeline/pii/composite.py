from ..base import PIIDetector, Entity

class CompositeDetector(PIIDetector):
    def __init__(self, *detectors: PIIDetector):
        self.detectors = detectors

    def detect(self, text: str) -> List[Entity]:
        all_ents = []
        for d in self.detectors:
            all_ents += d.detect(text)
        # 去重：如果 span 完全重疊，保 score 高者
        uniq = {}
        for e in all_ents:
            key = tuple(e["span"])
            if key not in uniq or e["score"] > uniq[key]["score"]:
                uniq[key] = e
        return sorted(uniq.values(), key=lambda x: x["span"][0])
