# src/deid_pipeline/pii/detectors/__init__.py
from pathlib import Path
import os
from .bert_detector import BertNERDetector
from .regex_detector import RegexDetector
from .bert_onnx_detector import BertONNXNERDetector
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
    cfg = Config()
    # 選擇主偵測器：ONNX > HF-BERT > spaCy
    use_onnx = cfg.USE_ONNX and cfg.ONNX_MODEL_PATH.exists()
    use_bert = not cfg.USE_STUB and cfg.NER_MODEL_PATH.exists()

    bert_cls = BertONNXNERDetector if use_onnx else BertNERDetector
    bert_path = (str(cfg.ONNX_MODEL_PATH) if use_onnx else str(cfg.NER_MODEL_PATH))

    detectors = []
    # 先嘗試 BERT／ONNX
    if use_bert:
        logger.info(f"使用 {'ONNX' if use_onnx else 'HF-BERT'} NER ({lang})")
        detectors.append(bert_cls(bert_path))

    # Regex 始終作為補漏
    regex_path = cfg.REGEX_RULES_FILE if lang == "zh" else cfg.REGEX_EN_RULES_FILE
    detectors.append(RegexDetector(regex_path))

    # 如果前面都沒加到主偵測器，就 fallback spaCy + Regex
    if not detectors or cfg.USE_STUB:
        logger.info(f"使用 spaCy 偵測 (備用方案 {lang})")
        detectors = [SpacyDetector(), RegexDetector(regex_path)]

    return CompositeDetector(*detectors)

