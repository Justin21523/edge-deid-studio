from .base import PIIDetector, Entity

class CompositeDetector(PIIDetector):
    def __init__(self, *detectors: PIIDetector):
        self.detectors = detectors

    def detect(self, text: str) -> list[Entity]:
        spans: list[Entity] = []
        for det in self.detectors:
            spans.extend(det.detect(text))
        # 去掉重疊 (保留分數高者)
        spans.sort(key=lambda e: (e["span"][0], -(e["span"][1]-e["span"][0])))
        merged = []
        for e in spans:
            if not merged or e["span"][0] >= merged[-1]["span"][1]:
                merged.append(e)
            elif e["score"] > merged[-1]["score"]:
                merged[-1] = e
        return merged
