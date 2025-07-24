# src/deid_pipeline/ocr_processor.py
import os
import time
import logging
import cv2
import numpy as np
from typing import Dict

from .parser.text_extractor import OCRProcessor as BaseOCRProcessor
from deid_pipeline.pii import get_detector
from deid_pipeline.pii.utils.replacer import Replacer

logger = logging.getLogger(__name__)

class OCRPIIProcessor:
    def __init__(self, lang: str = "zh"):
        self.ocr_processor = BaseOCRProcessor.get_instance()
        self.detector = get_detector(lang)
        self.replacer = Replacer()
        self.lang = lang

    def process_image(self, image_path: str) -> Dict:
        """處理圖像並返回 PII 掃描與替換結果"""
        start_time = time.perf_counter()
        result = {
            "status": "success",
            "original_text": "",
            "clean_text": "",
            "entities": [],
            "events": [],
            "processing_time": 0
        }

        try:
            # OCR 處理
            ocr_output = self.ocr_processor.process_image_file(image_path)
            result["original_text"] = ocr_output.get("text", "")

            # PII 檢測
            entities = self.detector.detect(result["original_text"])
            result["entities"] = entities

            # 替換
            clean_text, events = self.replacer.replace(
                result["original_text"], entities
            )
            result["clean_text"] = clean_text
            result["events"] = events

        except Exception as e:
            logger.error(f"OCRPII processing failed: {image_path}, error: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        finally:
            result["processing_time"] = time.perf_counter() - start_time
            logger.info(
                f"OCRPII complete: {image_path}, time: {result['processing_time']:.2f}s"
            )

        return result

# Extend BaseOCRProcessor to add image file support
class OCRProcessor(BaseOCRProcessor):
    def process_image_file(self, image_path: str) -> Dict:
        """讀取圖檔並回傳 OCR 細節"""
        self.init_reader()
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")

            results = self.reader.readtext(img)
            text = ""
            details = []
            for bbox, word, conf in results:
                text += word + " "
                details.append({
                    "text": word,
                    "confidence": float(conf),
                    "bbox": bbox
                })

            return {"text": text.strip(), "details": details}
        except Exception as e:
            logger.error(f"OCR processing failed on image: {e}")
            return {"text": "", "details": [], "error": str(e)}
