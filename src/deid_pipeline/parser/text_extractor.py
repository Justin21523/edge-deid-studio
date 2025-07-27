# src/deid_pipeline/parser/text_extractor.py
import os
import time
import fitz  # PyMuPDF
import numpy as np
import cv2
from bs4 import BeautifulSoup
from docx import Document
from .ocr import get_ocr_reader
from ..config import Config
from ..pii.utils import logger
import re

class TextExtractor:
    """統一的文字提取器，支援多種檔案格式"""
    def __init__(self, lang="zh", ocr_engine="auto"):
        self.lang = lang
        self.ocr_engine = ocr_engine
        self.ocr_processor = None

    def init_ocr(self):
        if self.ocr_processor is None:
            self.ocr_processor = get_ocr_reader(self.ocr_engine, self.lang)

    def extract_text(self, file_path: str) -> tuple[str, list]:
        """從檔案中提取文字並返回文字和偏移映射"""
        start_time = time.perf_counter()
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".txt":
                return self._extract_txt(file_path)
            elif ext == ".docx":
                return self._extract_docx(file_path)
            elif ext == ".html":
                return self._extract_html(file_path)
            elif ext == ".pdf":
                return self._extract_pdf(file_path)
            elif ext in (".jpg", ".jpeg", ".png", ".bmp"):
                return self._extract_image(file_path)
            else:
                raise ValueError(f"不支援的檔案格式: {ext}")
        except Exception as e:
            logger.error(f"文字提取失敗: {file_path}, 錯誤: {str(e)}")
            return "", []
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info(f"文字提取完成: {file_path}, 耗時: {elapsed:.2f}秒")

    def _extract_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        offset_map = [((0, 0, 0, 0), i) for i in range(len(text))]
        return text, offset_map

    def _extract_docx(self, file_path):
        doc = Document(file_path)
        full_text = []
        offset_map = []
        char_index = 0

        for para in doc.paragraphs:
            para_text = para.text + "\n"
            full_text.append(para_text)

            # 簡化偏移映射
            for i in range(len(para_text)):
                offset_map.append(((-1, -1, -1, -1), char_index + i))

            char_index += len(para_text)

        return "".join(full_text), offset_map

    def _extract_html(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")

        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text).strip()
        offset_map = [((0, 0, 0, 0), i) for i in range(len(text))]
        return text, offset_map

    def _extract_pdf(self, file_path):
        self.init_ocr()
        doc = fitz.open(file_path)
        full_text = []
        offset_map = []
        char_index = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = ""

            # 先嘗試提取文字
            text_blocks = page.get_text("blocks", sort=True)
            for block in text_blocks:
                if block[6] == 0:  # 文字塊
                    block_text = block[4].strip()
                    if block_text:
                        page_text += block_text + "\n"

                        # 創建偏移映射
                        for i in range(len(block_text)):
                            offset_map.append((
                                (page_num, block[0], block[1], block[2], block[3]),
                                char_index + i
                            ))
                        char_index += len(block_text) + 1

            # OCR回退機制
            if len(page_text.strip()) < Config.OCR_THRESHOLD:
                logger.info(f"頁面 {page_num} 觸發OCR回退機制")
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                # 轉換為BGR格式
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                ocr_text, ocr_blocks = self.ocr_processor.recognize(img)
                page_text = ocr_text + "\n"

                # 創建OCR偏移映射
                for block in ocr_blocks:
                    text_len = len(block['text'])
                    for i in range(text_len):
                        offset_map.append((
                            (page_num, block['left'], block['top'],
                             block['left'] + block['width'],
                             block['top'] + block['height']),
                            char_index + i
                        ))
                    char_index += text_len + 1

            full_text.append(page_text)

        return "\n".join(full_text), offset_map

    def _extract_image(self, file_path):
        self.init_ocr()
        text, text_blocks = self.ocr_processor.recognize(file_path)

        # 創建偏移映射
        offset_map = []
        char_index = 0
        for block in text_blocks:
            for i, char in enumerate(block['text']):
                offset_map.append((
                    (0, block['left'], block['top'],
                     block['left'] + block['width'],
                     block['top'] + block['height']),
                    char_index
                ))
                char_index += 1
            char_index += 1  # 空格分隔

        return text, offset_map
