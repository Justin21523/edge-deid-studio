from .bert_detector import BertNERDetector
from .regex_detector import RegexDetector
from .composite import CompositeDetector

def get_detector(lang: str = "en"):
    if lang == "zh":
        bert = BertNERDetector("models/bert_ner_zh_q")
        regex = RegexDetector()
        return CompositeDetector(bert, regex)
    else:
        from spacy_langdetect import load  # 你原來的英文 detector
        return load("en_core_web_sm")
