from .bert_detector import BertNERDetector
from .regex_detector import RegexDetector
from .legacy.spacy_detector import SpacyDetector
from .composite import CompositeDetector
from pathlib import Path

"""
中央設定：只要把 USE_STUB 改成 False，就會切到真 ONNX / SpaCy 模型。
"""
USE_STUB = True
PACKAGE_ROOT = Path(__file__).resolve().parent          # .../pii
PROJECT_ROOT = PACKAGE_ROOT.parent.parent               # .../deid_pipeline/..

MODEL_ZH = PROJECT_ROOT / "models" / "bert_ner_zh_q"    # 絕對路徑

def get_detector(lang="zh")-> CompositeDetector:
    if lang == "zh":
        return CompositeDetector(
            BertNERDetector(str(MODEL_ZH)),
            RegexDetector()
        )
    else:                        # en / default
        return CompositeDetector(SpacyDetector())

