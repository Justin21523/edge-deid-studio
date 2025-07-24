# src/deid_pipeline/config.py
import os
from pathlib import Path
import yaml

PROJECT_ROOT       = Path(__file__).resolve().parent.parent
CONFIGS_DIR        = PROJECT_ROOT / "configs"

# PII 規則檔
REGEX_RULES_FILE   = CONFIGS_DIR / "regex_zh.yaml"
# 你也可以加入 "regex_en.yaml" 等

def load_regex_rules(path: Path = REGEX_RULES_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

class Config:
    """Central configuration for text extraction, PII detection and fake data."""
    # --- paths & formats ---
    SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".png", ".jpg"]

    # --- text extraction ---
    OCR_ENABLED       = True
    OCR_THRESHOLD     = 50
    OCR_LANGUAGES     = ["ch_tra", "en"]

    # --- PII detection (BERT) ---
    NER_MODEL_PATH    = os.getenv("NER_MODEL_PATH", PROJECT_ROOT / "models" / "ner")
    BERT_CONFIDENCE_THRESHOLD = 0.85
    MAX_SEQ_LENGTH    = 512
    ENTITY_PRIORITY = {
        "TW_ID": 100,
        "PASSPORT": 95,
        "UNIFIED_BUSINESS_NO": 90,
        "PHONE": 85,
        "EMAIL": 80,
        "NAME": 75,
        "ADDRESS": 70,
    }
    WINDOW_STRIDE     = 0.5

    # --- regex rules ---
    REGEX_PATTERNS    = load_regex_rules()

    # --- fake data ---
    GPT2_MODEL_PATH   = os.getenv("GPT2_MODEL_PATH", PROJECT_ROOT / "models" / "gpt2")
    FAKER_LOCALE      = "zh_TW"
    FAKER_CACHE_SIZE  = 1000

    # --- ONNX runtime ---
    USE_ONNX          = True
    ONNX_MODEL_PATH   = os.getenv("ONNX_MODEL_PATH", PROJECT_ROOT / "edge_models" / "bert-ner-zh.onnx")
    ONNX_PROVIDERS    = ["CPUExecutionProvider", "CUDAExecutionProvider", "NPUExecutionProvider"]

    # --- logging & env ---
    ENVIRONMENT       = os.getenv("ENV", "local")
    LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PROFILING  = False

    # System flags
    USE_STUB = False
    LOG_LEVEL = "INFO"

    # ONNX 推論相關
    USE_ONNX          = True
    ONNX_MODEL_PATH   = Path(os.getenv("ONNX_MODEL_PATH", PROJECT_ROOT / "edge_models" / "bert-ner-zh.onnx"))
    ONNX_PROVIDERS    = ["CPUExecutionProvider", "CUDAExecutionProvider", "NPUExecutionProvider"]

    # 長文本分段
    MAX_SEQ_LENGTH    = 512
    WINDOW_STRIDE     = 0.5
