# src/deid_pipeline/pii/detectors/__init__.py
from pathlib import Path
import os
from .bert_detector import BertNERDetector
from .regex_detector import RegexDetector
from .composite import CompositeDetector
from .legacy.spacy_detector import SpacyDetector
from ...config import Config
from ..utils import logger

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
cfg = Config()
MODEL_ZH = cfg.NER_MODEL_PATH  # use central config
MODEL_EN = Path(os.getenv("NER_MODEL_PATH_EN", str(PROJECT_ROOT/"models"/"bert-ner-en")))

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
