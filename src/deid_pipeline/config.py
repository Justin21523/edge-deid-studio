# src/deid_pipeline/config.py
import os
from pathlib import Path

OCR_THRESHOLD = 50   # 少於 50 個字就用 OCR
USE_STUB = False     # 跟舊有 USE_STUB 保持同步

# 定義 GPT-2 模型路徑，環境變數優先，否則預設到 models/gpt2-base
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPT2_MODEL_PATH = os.getenv(
    "GPT2_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "gpt2-base")
)
