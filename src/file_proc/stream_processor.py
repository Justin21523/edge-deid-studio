import io
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
#from .image_ocr import OCRProcessor  # OCR 部分後面才整合
from docx import Document          # DOCX 部分後面才整合
import re

class FileParser:
    def __init__(self, file_bytes: bytes, file_type: str):
        self.file_bytes = io.BytesIO(file_bytes)
        self.file_type = file_type

    def process(self) -> dict:
        """統一處理入口，返回文字內容和圖片列表"""
        if self.file_type == 'pdf':
            return self._process_pdf()
        elif self.file_type == 'docx':
            return self._process_docx()
        else:
            #其他格式處理
            raise ValueError(f"Unsupported file type: {self.file_type}")

    def _process_pdf(self) -> dict:
        result = {'text': '', 'images': []}
        data = self.file_bytes.getvalue()
        try:
            with fitz.open(stream=data, filetype='pdf') as doc:
                for page in doc:
                    result['text'] += page.get_text()
            # 文字層少於 50 字，啟用影像轉文字
            if len(result['text']) < 50:
                images = convert_from_bytes(data)
                result['images'] = images  # 留給 OCR 模組處理
        except Exception:
            # 讀取失敗時直接把每頁轉成圖片
            result['images'] = convert_from_bytes(data)
        return result

    def _process_docx(self) -> dict:
        # 用 python-docx 直接讀文字
        self.file_bytes.seek(0)
        doc = Document(self.file_bytes)
        text = "\n".join([para.text for para in doc.paragraphs])
        return {'text': text, 'images': []}

    def _optimize_for_edge(self, content: str) -> str:
        # 1. 移除重複出現的頁眉／頁腳
        lines = content.split('\n')
        seen = {}
        filtered = []
        for line in lines:
            seen[line] = seen.get(line, 0) + 1
            if seen[line] <= 2:  # 最多保留前兩次
                filtered.append(line)
        content = "\n".join(filtered)
        # 2. 壓縮連續空白字元
        content = re.sub(r'\s{2,}', ' ', content)
        # 3. （可擴充）分段 yield 或避免記憶體爆
        return content
