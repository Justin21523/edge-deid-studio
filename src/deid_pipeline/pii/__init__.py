from pathlib import Path
from ..config import USE_STUB
from .bert_detector import BertNERDetector
from .regex_detector import RegexDetector
from .composite import CompositeDetector
from .legacy.spacy_detector import SpacyDetector
import logging

"""
中央設定：只要把 USE_STUB 改成 False，就會切到真 ONNX / SpaCy 模型。
"""
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
MODEL_ZH = PROJECT_ROOT / "models" / "bert_ner_zh_q"

MODEL_PATHS = {
    "zh": PROJECT_ROOT / "models" / "bert_ner_zh_q",
    "en": PROJECT_ROOT / "models" / "bert_ner_en",
    "ja": PROJECT_ROOT / "models" / "bert_ner_ja",
}

def get_detector(lang="zh")-> CompositeDetector:
    logger = logging.getLogger(__name__)
    logger.info(f"[get_detector] 取得 {lang} 檢測器組合")
    # 檢查模型路徑
    model_path = MODEL_PATHS.get(lang)

    if model_path and model_path.exists() and not USE_STUB:
        detector = CompositeDetector(
            BertNERDetector(str(model_path)),
            RegexDetector()
        )
    else:
        # 英文或其他語言
        # fallback to SpaCy + regex
        cfg = f"configs/regex_{lang}.yaml" if lang != "zh" else "configs/regex_zh.yaml"
        detector = CompositeDetector(
            SpacyDetector(),
            RegexDetector(config_path=cfg)
        )
    logger.info(f"[get_detector] 使用檢測器：{', '.join(d.__class__.__name__ for d in detector.detectors)}")
    return detector
