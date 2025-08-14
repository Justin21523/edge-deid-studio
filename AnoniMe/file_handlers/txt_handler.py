# file_handlers/text_handler.py
import os
from pii_models.presidio_detector import detect_pii
from faker_models.presidio_replacer import replace_pii

class TextHandler:
    """
    處理純文字格式 (.txt, .csv, .html, .json) in-place 去識別化，
    抽取純文字、替換 PII、再輸出純文字檔。
    """
    def deidentify(self, input_path: str, output_path: str, language: str = "auto") -> str:
        # 檢查檔案是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到輸入檔: {input_path}")

        # 1) 讀取純文字內容
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 2) 偵測 PII
        entities = detect_pii(text, language="en", score_threshold=0.6)
        
        # 3) 使用假資料或遮蔽進行替換
        cleaned = replace_pii(text, entities)

        # 4) 寫入輸出檔
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        return output_path


