from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...config import Config
from ..utils import logger

from .composite import CompositeDetector
from .regex_detector import RegexDetector

if TYPE_CHECKING:  # pragma: no cover
    from ..utils.base import PIIDetector


def _model_dir_for_lang(cfg: Config, lang: str) -> Path:
    return cfg.NER_MODEL_PATH_ZH if lang == "zh" else cfg.NER_MODEL_PATH_EN


def get_detector(lang: str = "zh") -> CompositeDetector:
    """Create a composite detector based on configuration and local availability.

    Design goals:
    - Default to local-only, lightweight detection (regex) when models are unavailable.
    - Avoid importing heavy optional dependencies at module import time.
    - Prefer ONNX over PyTorch when explicitly enabled and the model exists locally.
    """

    cfg = Config()
    detectors: list["PIIDetector"] = []

    regex_path = cfg.REGEX_RULES_FILE if lang == "zh" else cfg.REGEX_EN_RULES_FILE

    # Prefer ONNX if enabled and available locally.
    if cfg.USE_ONNX and cfg.ONNX_MODEL_PATH.exists() and not cfg.USE_STUB:
        try:
            from .bert_onnx_detector import BertONNXNERDetector

            model_dir = _model_dir_for_lang(cfg, lang)
            detectors.append(
                BertONNXNERDetector(
                    onnx_model_path=cfg.ONNX_MODEL_PATH,
                    tokenizer_dir=model_dir,
                    providers=cfg.ONNX_PROVIDERS,
                )
            )
            logger.info("Using ONNX NER detector (lang=%s)", lang)
        except Exception as exc:
            logger.warning("Failed to initialize ONNX NER detector: %s", exc)

    # Fallback to PyTorch/Transformers BERT if enabled.
    if not detectors and not cfg.USE_STUB:
        model_dir = _model_dir_for_lang(cfg, lang)
        if model_dir.exists():
            try:
                from .bert_detector import BertNERDetector

                detectors.append(BertNERDetector(model_dir))
                logger.info("Using HF Transformers NER detector (lang=%s)", lang)
            except Exception as exc:
                logger.warning("Failed to initialize HF NER detector: %s", exc)

    # Optional spaCy fallback (must be explicitly enabled).
    if cfg.USE_SPACY:
        try:
            from .legacy.spacy_detector import SpacyDetector

            detectors.append(SpacyDetector(lang=lang))
            logger.info("Using spaCy fallback detector (lang=%s)", lang)
        except Exception as exc:
            logger.warning("Failed to initialize spaCy detector: %s", exc)

    # Regex is always enabled as a recall backstop.
    detectors.append(RegexDetector(regex_path))

    return CompositeDetector(*detectors)
