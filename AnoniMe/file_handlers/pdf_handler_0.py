# file_handlers/pdf_pii_detector.py
import os
import fitz  # PyMuPDF
from pii_models.presidio_detector import detect_pii

class PdfPiiDetector:
    """
    從 PDF 中抽取文字並偵測 PII，回傳可閱讀的列表。
    每個實體包含：頁碼、實體類型、起訖位置、匹配文字。
    """
    def extract_pii(self, input_path: str, language: str = "auto"):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到輸入檔: {input_path}")
        results = []
        # 開啟 PDF
        doc = fitz.open(input_path)
        # 逐頁處理
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            # 偵測 PII
            entities = detect_pii(text, language="en", score_threshold=0.6)
            # 收集結果
            for ent in entities:
                snippet = text[ent.start:ent.end]
                results.append({
                    "page": page_num,
                    "entity_type": ent.entity_type,
                    "start": ent.start,
                    "end": ent.end,
                    "text": snippet
                })
        return results