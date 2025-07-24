import os
import json
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup
from docx import Document
import fitz  # PyMuPDF
import easyocr
from deid_pipeline.parser.ocr import get_ocr_reader
from deid_pipeline.config import OCR_THRESHOLD, USE_STUB, Config
from deid_pipeline.pii.utils import logger

# 全域OCR處理器
class OCRProcessor:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.reader = None

    def init_reader(self):
        if self.reader is None:
            self.reader = easyocr.Reader(["ch_tra", "en"], gpu=True)
            logger.info("EasyOCR閱讀器已初始化")

    def process_page(self, pix):
        self.init_reader()
        try:
            # 將PyMuPDF的pixmap轉換為numpy陣列
            samples = pix.samples
            h, w = pix.height, pix.width
            img = np.frombuffer(samples, dtype=np.uint8).reshape((h, w, pix.n))

            # 處理圖像格式
            if pix.n == 4:  # RGBA轉換為RGB
                img = img[..., :3]

            results = self.reader.readtext(img)
            return "\n".join(res[1] for res in results)
        except Exception as e:
            logger.error(f"OCR處理失敗: {str(e)}")
            return ""

# only text will be extracted!
def extract_text(file_path: str, ocr_fallback: bool = True) -> tuple[str, list]:
    """從文件中提取文字並返回文字和偏移映射"""
    start_time = time.perf_counter()
    ext = os.path.splitext(file_path)[1].lower()
    offset_map = []
    current_index = 0

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            # 創建簡單的偏移映射
            for i in range(len(text)):
                offset_map.append(((0, 0, 0, 0, 0), i))  # (page, block_x0, block_y0, block_x1, block_y1)

            return text, offset_map

        elif ext == ".docx":
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
                # 文檔偏移映射較複雜，此處簡化處理
                for i in range(len(para.text) + 1):
                    offset_map.append(((-1, -1, -1, -1, -1), current_index + i))
                current_index += len(para.text) + 1
            return text, offset_map

        elif ext == ".html":
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")

            # 簡化偏移映射
            for i in range(len(text)):
                offset_map.append(((-1, -1, -1, -1, -1), i))

            return text, offset_map

        elif ext == ".pdf":
            doc = fitz.open(file_path)
            full_text = []
            ocr_processor = OCRProcessor.get_instance()

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("blocks", sort=True)
                page_text = ""

                for block in blocks:
                    if block[6] == 0:  # 僅處理文字區塊
                        text = block[4].strip()
                        if text:
                            # 創建偏移映射
                            for i, char in enumerate(text):
                                offset_map.append((
                                    (page_num, block[0], block[1], block[2], block[3]),
                                    current_index + i
                                ))

                            page_text += text + "\n"
                            current_index += len(text) + 1

                # OCR回退機制
                if ocr_fallback and len(page_text.strip()) < Config.OCR_THRESHOLD:
                    logger.info(f"頁面 {page_num} 觸發OCR回退機制")
                    pix = page.get_pixmap()
                    ocr_text = ocr_processor.process_page(pix)
                    page_text = ocr_text + "\n"
                    current_index += len(ocr_text) + 1

                    # 更新偏移映射
                    for i, char in enumerate(ocr_text):
                        offset_map.append((
                            (page_num, 0, 0, pix.width, pix.height),
                            current_index - len(ocr_text) + i - 1
                        ))

                full_text.append(page_text)

            return "\n".join(full_text), offset_map

        else:
            raise ValueError(f"不支援的檔案格式: {ext}")

    except Exception as e:
        logger.error(f"文字提取失敗: {file_path}, 錯誤: {str(e)}")
        return "", []

    finally:
        elapsed = time.perf_counter() - start_time
        logger.info(f"文字提取完成: {file_path}, 耗時: {elapsed:.2f}秒")
