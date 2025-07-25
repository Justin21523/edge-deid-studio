# src/deid_pipeline/image_deid/processor.py
import time
import cv2
from typing import Dict

from deid_pipeline.parser.ocr import get_ocr_reader
from deid_pipeline.pii.detectors import get_detector
from deid_pipeline.pii.utils.replacer import Replacer
from deid_pipeline.config import Config

class ImageDeidProcessor:
    def __init__(self, lang: str = "zh"):
        # EasyOCR 讀取器
        self.reader   = get_ocr_reader(Config.OCR_LANGUAGES)
        # PII 偵測：BERT + Regex composite
        self.detector = get_detector(lang)
        self.replacer = Replacer()
        self.lang     = lang

    def process_image(self, image_path: str) -> Dict:
        start = time.perf_counter()
        img   = cv2.imread(image_path)
        # OCR 擷取文字
        raw_results = self.reader.readtext(img)
        text_lines  = [res[1] for res in raw_results]
        original_text = "\n".join(text_lines)

        # PII 偵測
        entities = self.detector.detect(original_text)

        # 取代或遮蔽
        clean_text, events = self.replacer.replace(original_text, entities)

        return {
            "status": "success",
            "original_text": original_text,
            "clean_text":    clean_text,
            "entities":      entities,
            "events":        events,
            "processing_time": time.perf_counter() - start
        }
