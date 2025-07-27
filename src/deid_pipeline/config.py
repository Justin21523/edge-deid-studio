# src/deid_pipeline/config.py
import os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR  = PROJECT_ROOT / "configs"


def load_regex_rules(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
class Config:
    """Central configuration for text extraction, PII detection and fake data."""
    # PII 規則檔，可透過環境變數覆寫
    REGEX_RULES_FILE    = Path(os.getenv("REGEX_RULES_FILE",
                                          CONFIGS_DIR/"regex_zh.yaml"))
    REGEX_EN_RULES_FILE = Path(os.getenv("REGEX_EN_RULES_FILE",
                                          CONFIGS_DIR/"regex_en.yaml"))
    # --- paths & formats ---
    # 支援的檔案格式：文件、試算表、簡報、影像、純文字、HTML
    SUPPORTED_FILE_TYPES = [
        ".pdf",     # PDF 文件
        ".docx",    # Word 文件
        ".xlsx",    # Excel 試算表
        ".pptx",    # PowerPoint 簡報
        ".txt",     # 純文字
        ".html",    # HTML
        ".png",     # 圖像
        ".jpg",     # 圖像
        ".jpeg",    # 圖像
    ]

    # --- text extraction ---
    OCR_ENABLED       = True
    OCR_THRESHOLD     = 50
    OCR_LANGUAGES     = ["ch_tra", "en"]

    # --- PII detection (BERT) ---
    NER_MODEL_PATH    = os.getenv("NER_MODEL_PATH", PROJECT_ROOT / "models" / "ner")
    BERT_CONFIDENCE_THRESHOLD = 0.85
    MAX_SEQ_LENGTH    = 512
    ENTITY_PRIORITY = {
        "ID":               100,
        "PASSPORT":        95,
        "UNIFIED_BUSINESS_NO": 90,
        "PHONE":           85,
        "EMAIL":           80,
        "NAME":            75,
        "ADDRESS":         70,
        "ORGANIZATION":    65,
        "MEDICAL_ID":      60,
    }
    WINDOW_STRIDE     = 0.5

    # --- regex rules ---
    REGEX_PATTERNS    = load_regex_rules(REGEX_RULES_FILE)

    # --- fake data ---
    GPT2_MODEL_PATH   = os.getenv("GPT2_MODEL_PATH", PROJECT_ROOT / "models" / "gpt2")
    FAKER_LOCALE      = "zh_TW"
    FAKER_CACHE_SIZE  = 1000

    # --- ONNX runtime ---
    USE_ONNX          = False
    ONNX_MODEL_PATH   = Path(os.getenv("ONNX_MODEL_PATH", PROJECT_ROOT / "edge_models" / "bert-ner-zh.onnx"))
    ONNX_PROVIDERS    = ["CPUExecutionProvider","CUDAExecutionProvider","NPUExecutionProvider"]

    # --- logging & env ---
    ENVIRONMENT       = os.getenv("ENV", "local")
    LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PROFILING  = False
    USE_STUB          = True

    # --- 長文本分段 ---
    MAX_SEQ_LENGTH    = 512
    WINDOW_STRIDE     = 0.5
