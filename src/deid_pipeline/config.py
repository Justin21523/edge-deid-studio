from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .storage.layout import StorageLayout


def find_project_root(start: Path | None = None) -> Path:
    """Resolve the repository root directory.

    The project root is used to locate `configs/`, `models/`, and other repo-relative
    assets during local development. For production deployments, prefer overriding
    paths via environment variables.
    """

    env_root = os.getenv("EDGE_DEID_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    start_path = (start or Path(__file__)).resolve()
    for parent in [start_path] + list(start_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / "configs").exists():
            return parent

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
STORAGE = StorageLayout.from_project_root(PROJECT_ROOT)


def load_regex_rules(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Config:
    """Central configuration for extraction, detection, and replacement."""

    # --- Paths ---
    REGEX_RULES_FILE = Path(
        os.getenv("REGEX_RULES_FILE", str(CONFIGS_DIR / "regex_zh.yaml"))
    )
    REGEX_EN_RULES_FILE = Path(
        os.getenv("REGEX_EN_RULES_FILE", str(CONFIGS_DIR / "regex_en.yaml"))
    )

    # Default model locations (override via env vars in production).
    # These defaults follow the AI_WAREHOUSE 3.0 layout when available:
    # - models: /mnt/c/ai_models
    # - datasets/training outputs: /mnt/data
    NER_MODEL_PATH_ZH = Path(
        os.getenv(
            "NER_MODEL_PATH_ZH",
            str(STORAGE.edge_deid_models_home / "bert-ner-zh"),
        )
    )
    NER_MODEL_PATH_EN = Path(
        os.getenv(
            "NER_MODEL_PATH_EN",
            str(STORAGE.edge_deid_models_home / "bert-ner-en"),
        )
    )
    GPT2_MODEL_PATH = Path(
        os.getenv("GPT2_MODEL_PATH", str(STORAGE.models_home / "llm" / "gpt2"))
    )
    ONNX_MODEL_PATH = Path(
        os.getenv(
            "ONNX_MODEL_PATH",
            str(STORAGE.edge_deid_models_home / "bert-ner-zh.onnx"),
        )
    )

    # --- Supported file types (handler-based in future refactor) ---
    SUPPORTED_FILE_TYPES = [
        ".pdf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".csv",
        ".txt",
        ".html",
        ".png",
        ".jpg",
        ".jpeg",
    ]

    # --- OCR ---
    OCR_ENGINE = os.getenv("OCR_ENGINE", "auto")  # "tesseract", "easyocr", or "auto"
    OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "60"))
    OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD", "50"))  # OCR fallback min chars
    OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in {"1", "true", "yes"}
    USE_GPU = os.getenv("USE_GPU", "false").lower() in {"1", "true", "yes"}

    # --- NER / detection ---
    BERT_CONFIDENCE_THRESHOLD = float(os.getenv("BERT_CONFIDENCE_THRESHOLD", "0.85"))
    MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
    WINDOW_STRIDE = float(os.getenv("WINDOW_STRIDE", "0.5"))
    USE_ONNX = os.getenv("USE_ONNX", "false").lower() in {"1", "true", "yes"}
    _ONNX_PROVIDERS_ENV = os.getenv("ONNX_PROVIDERS", "").strip()
    if _ONNX_PROVIDERS_ENV:
        ONNX_PROVIDERS = [p.strip() for p in _ONNX_PROVIDERS_ENV.split(",") if p.strip()]
    else:
        # Prefer GPU providers when USE_GPU=true; `select_onnx_providers` will filter to the
        # locally available set at runtime.
        ONNX_PROVIDERS = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if USE_GPU
            else ["CPUExecutionProvider"]
        )
        # Keep experimental providers last as optional fallbacks.
        ONNX_PROVIDERS.append("NPUExecutionProvider")

    # Optional: allow spaCy fallback if explicitly enabled.
    USE_SPACY = os.getenv("USE_SPACY", "false").lower() in {"1", "true", "yes"}

    # --- Replacement / fake data ---
    FAKER_LOCALE = os.getenv("FAKER_LOCALE", "zh_TW")
    FAKER_CACHE_SIZE = int(os.getenv("FAKER_CACHE_SIZE", "1000"))

    # --- Conflict resolution ---
    ENTITY_PRIORITY = {
        "ID": 100,
        "PASSPORT": 95,
        "PHONE": 90,
        "UNIFIED_BUSINESS_NO": 85,
        "EMAIL": 80,
        "NAME": 75,
        "ADDRESS": 70,
        "ORGANIZATION": 65,
        "MEDICAL_ID": 60,
    }

    # --- Runtime ---
    ENVIRONMENT = os.getenv("ENV", "local")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() in {"1", "true", "yes"}
    USE_STUB = os.getenv("USE_STUB", "true").lower() in {"1", "true", "yes"}

    # Backward-compatible convenience: keep regex patterns available if the file exists.
    try:
        REGEX_PATTERNS = load_regex_rules(REGEX_RULES_FILE)
    except FileNotFoundError:
        REGEX_PATTERNS: dict[str, Any] = {}
