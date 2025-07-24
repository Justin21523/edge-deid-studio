from pathlib import Path
from .bert_detector import BertNERDetector
from .regex_detector import RegexDetector
from .composite import CompositeDetector
from .legacy.spacy_detector import SpacyDetector
from deid_pipeline.config import Config
import logging


logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
MODEL_ZH = PROJECT_ROOT / "models" / "bert_ner_zh_q"
MODEL_EN = PROJECT_ROOT / "models" / "bert_ner_en"

def get_detector(lang: str = "zh") -> CompositeDetector:
    config = Config()

    try:
        if lang == "zh" and not config.USE_STUB and MODEL_ZH.exists():
            logger.info("創建繁體中文檢測器 (BERT + Regex)")
            return CompositeDetector(
                BertNERDetector(str(MODEL_ZH)),
                RegexDetector()
            )
        elif lang == "en" and not config.USE_STUB and MODEL_EN.exists():
            logger.info("創建英文檢測器 (BERT + Regex)")
            return CompositeDetector(
                BertNERDetector(str(MODEL_EN)),
                RegexDetector(config_path="configs/regex_en.yaml")
            )
    except Exception as e:
        logger.error(f"創建BERT檢測器失敗，使用備用方案: {str(e)}")

    # 備用檢測器
    if lang == "zh":
        logger.info("使用備用中文檢測器 (SpaCy + Regex)")
        return CompositeDetector(
            SpacyDetector(),
            RegexDetector()
        )
    else:
        logger.info("使用備用英文檢測器 (SpaCy + Regex)")
        return CompositeDetector(
            SpacyDetector(),
            RegexDetector(config_path="configs/regex_en.yaml")
        )
