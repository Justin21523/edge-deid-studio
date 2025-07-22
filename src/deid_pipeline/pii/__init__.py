from .bert_detector import BertNERDetector
from .regex_detector import RegexDetector
from .legacy.spacy_detector import SpacyDetector
from .composite import CompositeDetector

def get_detector(lang="zh")-> CompositeDetector:
    if lang == "zh":
        return CompositeDetector(
            BertNERDetector("models/bert_ner_zh_q"),
            RegexDetector()
        )
    else:                        # en / default
        return CompositeDetector(SpacyDetector())
