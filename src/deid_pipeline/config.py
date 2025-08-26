# src/deid_pipeline/config.py
import os
from pathlib import Path
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_regex_rules(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
class Config:
    """Central configuration for text extraction, PII detection and fake data."""
    # PII 規則檔，可透過環境變數覆寫
    REGEX_RULES_FILE    = Path(os.getenv("REGEX_RULES_FILE", PROJECT_ROOT / "configs" / "regex_zh.yaml"))
    REGEX_EN_RULES_FILE = Path(os.getenv("REGEX_EN_RULES_FILE", PROJECT_ROOT / "configs" /"regex_en.yaml"))
    # --- regex rules ---
    REGEX_PATTERNS    = load_regex_rules(REGEX_RULES_FILE)

    # --- logging & env ---
    ENVIRONMENT       = os.getenv("ENV", "local")
    LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PROFILING  = False
    USE_STUB          = True

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

    # ======  PII detection BERT NER 配置 ======
    # 模型相關
    BERT_MODEL_PATH    = os.getenv("BERT_MODEL_PATH", PROJECT_ROOT / "models" / "ner")
    BERT_DEFAULT_MODEL = "bert-base-chinese-ner"
    BERT_CONFIDENCE_THRESHOLD = float(os.getenv("BERT_CONFIDENCE_THRESHOLD", "0.7"))
    BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "512"))
    BERT_BATCH_SIZE = int(os.getenv("BERT_BATCH_SIZE", "8"))

    MAX_SEQ_LENGTH    = 512
    WINDOW_STRIDE     = 0.5

    # --- ONNX runtime 配置---
    USE_ONNX          = False
    ONNX_MODEL_PATH   = Path(os.getenv("ONNX_MODEL_PATH", PROJECT_ROOT / "edge_models" / "bert-ner-zh.onnx"))
    ONNX_PROVIDERS    = ["CPUExecutionProvider","CUDAExecutionProvider","NPUExecutionProvider"]


    # 偵測器優先級 (數字越小優先級越高)
    ENTITY_PRIORITY = {
        "ID_NUMBER": 1,
        "PHONE_NUMBER": 2,
        "EMAIL_ADDRESS": 3,
        "PERSON": 4,
        "ADDRESS": 5,
        "ORGANIZATION": 6,
        "DATE": 7
    }

    # 偵測器權重配置
    DETECTOR_WEIGHTS = {
        "regex": 0.8,
        "bert_onnx": 1.3,
        "bert_pytorch": 1.2,
        "spacy": 0.9
    }

    # ====== OCR 配置 ======
    OCR_ENGINE = os.getenv("OCR_ENGINE", "auto")  # "tesseract", "easyocr", "auto"
    OCR_LANGUAGES     = ["ch_tra", "en"]
    OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.5"))
    OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", "100"))  # PDF 文字量閾值 當頁面文字少於此值時觸發OCR回退
    OCR_ENABLED       = True

    # OCR 前處理參數
    DESKEW_MIN_ANGLE = float(os.getenv("DESKEW_MIN_ANGLE", "0.5"))
    CLAHE_CLIP_LIMIT = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
    GAUSSIAN_BLUR_KERNEL = int(os.getenv("GAUSSIAN_BLUR_KERNEL", "1"))

    # 版面分析配置
    USE_LAYOUT_MODEL = os.getenv("USE_LAYOUT_MODEL", "true").lower() == "true"
    LAYOUT_MODEL_PATH = os.getenv("LAYOUT_MODEL_PATH", PROJECT_ROOT / "models" / "layout" / "layoutlmv3.onnx")

    # OCR 引擎權重 (融合模式使用)
    OCR_ENGINE_WEIGHTS = {
        "tesseract": 0.6,
        "easyocr": 0.4
    }

    # ====== 替換器配置 ======
    REPLACEMENT_MODE = os.getenv("REPLACEMENT_MODE", "mask")  # "mask" or "fake"
    FAKE_DATA_CONSISTENCY = True  # 保持假資料一致性


    # GPT-2 假資料生成配置
    GPT2_MODEL_PATH = os.getenv("GPT2_MODEL_PATH", PROJECT_ROOT / "models" / "gpt2")
    GPT2_MAX_LENGTH = int(os.getenv("GPT2_MAX_LENGTH", "50"))
    GPT2_TEMPERATURE = float(os.getenv("GPT2_TEMPERATURE", "0.8"))
    FAKER_LOCALE      = "zh_TW"
    FAKER_CACHE_SIZE  = 1000

    # ====== 效能配置 ======
    # 處理並行度
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  # 文字分塊大小

    # 記憶體管理
    ENABLE_MEMORY_OPTIMIZATION = True
    CLEAR_CACHE_THRESHOLD = int(os.getenv("CLEAR_CACHE_THRESHOLD", "100"))  # 處理多少個檔案後清理快取

    # ====== 日誌配置 ======
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ENABLE_DEBUG_OUTPUT = os.getenv("ENABLE_DEBUG_OUTPUT", "false").lower() == "true"

    # ====== 檔案處理配置 ======
    # 支援的檔案格式
    SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"]
    SUPPORTED_DOCUMENT_FORMATS = [".txt", ".docx", ".csv", ".xlsx", ".html"]

    # 處理限制
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_IMAGE_RESOLUTION = (4000, 4000)  # (width, height)

    # ====== 輸出配置 ======
    OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "json")  # "json", "csv", "xml"
    SAVE_INTERMEDIATE_RESULTS = os.getenv("SAVE_INTERMEDIATE_RESULTS", "false").lower() == "true"
    ANNOTATION_COLORS = {
        "PERSON": (0, 255, 0),       # 綠色
        "ID_NUMBER": (255, 0, 0),    # 紅色
        "PHONE_NUMBER": (0, 0, 255), # 藍色
        "EMAIL_ADDRESS": (255, 255, 0),  # 黃色
        "ADDRESS": (255, 0, 255),    # 洋紅
        "ORGANIZATION": (0, 255, 255), # 青色
        "DATE": (128, 128, 128),     # 灰色
        "default": (64, 64, 64)      # 深灰
    }

    # ====== 模型下載配置 ======
    MODEL_DOWNLOAD_CONFIG = {
        "bert-base-chinese-ner": {
            "hf_model_name": "ckiplab/bert-base-chinese-ner",
            "local_path": BERT_MODEL_PATH / "bert-base-chinese-ner",
            "onnx_optimize": True,
            "quantize": False  # 是否進行量化
        },
        "bert-base-multilingual-ner": {
            "hf_model_name": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "local_path": BERT_MODEL_PATH / "bert-base-multilingual-ner",
            "onnx_optimize": True,
            "quantize": False
        }
    }

    # ====== 驗證和測試配置 ======
    # 測試資料路徑
    TEST_DATA_DIR = Path(os.getenv("TEST_DATA_DIR", PROJECT_ROOT / "data" /"test"))

    BENCHMARK_TEXTS = [
        "我的姓名是王小明，身分證字號是A123456789。",
        "請聯絡張三先生，電話：02-1234-5678，信箱：zhang@example.com",
        "地址：台北市信義區信義路五段7號101樓"
    ]

    # 效能基準
    PERFORMANCE_TARGETS = {
        "bert_onnx_inference_time_ms": 150,  # 單份文件推理時間目標
        "total_processing_time_ms": 500,    # 完整處理時間目標
        "memory_usage_mb": 1000             # 記憶體使用目標
    }

    # ====== 錯誤處理配置 ======
    ENABLE_FALLBACK_DETECTION = True  # 啟用降級偵測
    FALLBACK_HIERARCHY = [
        "bert_onnx",
        "bert_pytorch",
        "spacy",
        "regex"
    ]

    # 重試配置
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY_SECONDS = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))

    # ====== 硬體加速配置 ======
    # Snapdragon NPU 配置 (如果可用)
    ENABLE_NPU = os.getenv("ENABLE_NPU", "false").lower() == "true"
    NPU_PROVIDER_OPTIONS = {
        "device_id": 0,
        "backend": "qnn",  # Qualcomm Neural Network SDK
        "profiling": False
    }

    # CUDA 配置
    CUDA_DEVICE_ID = int(os.getenv("CUDA_DEVICE_ID", "0"))
    ENABLE_CUDA_OPTIMIZATION = True

    # ====== 快取配置 ======
    ENABLE_MODEL_CACHE = True
    CACHE_DIR = Path("cache")
    MODEL_CACHE_SIZE_MB = int(os.getenv("MODEL_CACHE_SIZE_MB", "2048"))

    # 結果快取
    ENABLE_RESULT_CACHE = os.getenv("ENABLE_RESULT_CACHE", "true").lower() == "true"
    RESULT_CACHE_TTL_HOURS = int(os.getenv("RESULT_CACHE_TTL_HOURS", "24"))

    # ====== 安全配置 ======
    # 隱私保護
    ENABLE_AUDIT_LOG = os.getenv("ENABLE_AUDIT_LOG", "true").lower() == "true"
    AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", PROJECT_ROOT / "logs" / "audit.log"))

    # 資料清理
    AUTO_CLEANUP_TEMP_FILES = True
    TEMP_FILE_RETENTION_HOURS = int(os.getenv("TEMP_FILE_RETENTION_HOURS", "2"))

    # --- 長文本分段 ---
    MAX_SEQ_LENGTH    = 512
    WINDOW_STRIDE     = 0.5

    EXTRACTOR_PLUGINS = {
        '.pdf': 'src.deid_pipeline.parser.text_extractor.PDFTextExtractor',
        '.docx': 'src.deid_pipeline.parser.text_extractor.DocxTextExtractor',
        '.html': 'src.deid_pipeline.parser.text_extractor.HTMLTextExtractor',
    }

    logger = logging.getLogger(__name__)


# ====== 整合測試配置 ======
def validate_config():
    """驗證配置有效性"""
    errors = []
    # 檢查必要目錄
    required_dirs = [Config.BERT_MODEL_PATH, Config.CACHE_DIR, Config.TEST_DATA_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists() and not Config.USE_STUB:
            dir_path.mkdir(parents=True, exist_ok=True)

    # 檢查模型檔案 (非 STUB 模式)
    if not Config.USE_STUB:
        for model_name, config in Config.MODEL_DOWNLOAD_CONFIG.items():
            model_path = config["local_path"]
            if not model_path.exists():
                errors.append(f"模型不存在: {model_path}")

    # 檢查數值範圍
    if not 0 < Config.BERT_CONFIDENCE_THRESHOLD <= 1:
        errors.append("BERT_CONFIDENCE_THRESHOLD 必須在 (0, 1] 範圍內")

    if Config.BERT_MAX_LENGTH <= 0:
        errors.append("BERT_MAX_LENGTH 必須大於 0")

    if errors:
        raise ValueError(f"配置驗證失敗: {'; '.join(errors)}")

def get_detector_config(detector_type: str) -> dict:
    """獲取指定偵測器的配置"""
    base_config = {
        "confidence_threshold": Config.BERT_CONFIDENCE_THRESHOLD,
        "max_length": Config.BERT_MAX_LENGTH,
        "batch_size": Config.BERT_BATCH_SIZE
    }

    if detector_type == "bert_onnx":
        base_config.update({
            "providers": Config.ONNX_PROVIDERS,
            "model_name": Config.BERT_DEFAULT_MODEL
        })
    elif detector_type == "bert_pytorch":
        base_config.update({
            "model_name": Config.BERT_DEFAULT_MODEL,
            "device": "cuda" if Config.ENABLE_CUDA_OPTIMIZATION else "cpu"
        })

    return base_config

def get_ocr_config() -> dict:
    """獲取 OCR 配置"""
    return {
        "engine": Config.OCR_ENGINE,
        "min_confidence": Config.OCR_MIN_CONFIDENCE,
        "threshold": Config.OCR_THRESHOLD,
        "deskew_min_angle": Config.DESKEW_MIN_ANGLE,
        "clahe_clip_limit": Config.CLAHE_CLIP_LIMIT,
        "gaussian_blur_kernel": Config.GAUSSIAN_BLUR_KERNEL,
        "use_layout_model": Config.USE_LAYOUT_MODEL,
        "layout_model_path": Config.LAYOUT_MODEL_PATH,
        "engine_weights": Config.OCR_ENGINE_WEIGHTS
    }

def get_processing_config() -> dict:
    """獲取處理配置"""
    return {
        "replacement_mode": Config.REPLACEMENT_MODE,
        "max_workers": Config.MAX_WORKERS,
        "chunk_size": Config.CHUNK_SIZE,
        "enable_memory_optimization": Config.ENABLE_MEMORY_OPTIMIZATION,
        "enable_fallback": Config.ENABLE_FALLBACK_DETECTION,
        "fallback_hierarchy": Config.FALLBACK_HIERARCHY
    }

# 初始化時驗證配置
if __name__ == "__main__":
    try:
        validate_config()
        print("配置驗證通過")
    except ValueError as e:
        print(f"配置錯誤: {e}")
        exit(1)
