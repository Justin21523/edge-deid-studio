# src/deid_pipeline/config.py
import os
from pathlib import Path

# 定義 GPT-2 模型路徑，環境變數優先，否則預設到 models/gpt2-base
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPT2_MODEL_PATH = os.getenv(
    "GPT2_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "gpt2-base")
)

class Config:
    """Central configuration for text extraction, PII detection and fake data."""

    # Text extraction
    OCR_ENABLED = True
    OCR_THRESHOLD = 50

    # PII detection
    BERT_CONFIDENCE_THRESHOLD = 0.85
    ENTITY_PRIORITY = {
        "TW_ID": 100,
        "PASSPORT": 95,
        "UNIFIED_BUSINESS_NO": 90,
        "PHONE": 85,
        "EMAIL": 80,
        "NAME": 75,
        "ADDRESS": 70,
    }

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    # Fake data generation
    GPT2_MODEL_PATH =  os.getenv(
        "GPT2_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "gpt2-base")
    )
    FAKER_LOCALE = "zh_TW"

    # System flags
    USE_STUB = False
    LOG_LEVEL = "INFO"

# backward‐compatibility for module imports
OCR_THRESHOLD = Config.OCR_THRESHOLD # 少於 50 個字就用 OCR
USE_STUB = Config.USE_STUB  # 跟舊有 USE_STUB 保持同步
