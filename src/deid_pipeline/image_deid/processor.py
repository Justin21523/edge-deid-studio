# src/deid_pipeline/image_deid/processor.py
import time
import os
from typing import Dict
import cv2
import numpy as np
from ..parser.text_extractor import SmartTextExtractor
from ..pii.detectors import get_detector
from ..pii.utils.replacer import Replacer
from ..config import Config

class ImageDeidProcessor:
    def __init__(self, config: Config = None, lang: str = "zh"):
        self.config = config or Config()
        # 文字提取器（內含OCR）
        self.extractor = SmartTextExtractor(config=self.config)
        # PII 偵測器
        self.detector = get_detector(lang)
        # 替換器
        self.replacer = Replacer()
        self.lang = lang

    def process(self, file_path: str) -> Dict:
        start = time.perf_counter()

        # 文字提取
        original_text, offset_map = self.extractor.extract(file_path)

        if not original_text.strip():
            return {
                "status": "error",
                "message": "無法提取文字內容",
                "processing_time": time.perf_counter() - start
            }

        # PII 偵測
        entities = self.detector.detect(original_text)

        # 取代或遮蔽
        clean_text, events = self.replacer.replace(original_text, entities)

        # 準備視覺化結果（僅對圖像檔案）
        visual_result = None
        if os.path.splitext(file_path)[1].lower() in ['.jpg', '.jpeg', '.png']:
            visual_result = self._generate_visual_result(file_path, entities, offset_map)

        return {
            "status": "success",
            "original_text": original_text,
            "clean_text": clean_text,
            "entities": entities,
            "events": events,
            "visual_result": visual_result,
            "processing_time": time.perf_counter() - start
        }

    def _generate_visual_result(self, image_path, entities, offset_map):
        """生成標註PII的視覺化結果"""
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 建立位置索引
        position_index = {pos[1]: pos[0] for pos in offset_map}

        for entity in entities:
            start, end = entity["span"]

            # 收集所有相關位置
            bboxes = []
            for i in range(start, end):
                if i in position_index:
                    page, left, top, right, bottom = position_index[i]
                    if page == 0:  # 只處理第一頁（單圖像）
                        bboxes.append((left, top, right, bottom))

            if not bboxes:
                continue

            # 計算合併邊界框
            all_left = min(b[0] for b in bboxes)
            all_top = min(b[1] for b in bboxes)
            all_right = max(b[2] for b in bboxes)
            all_bottom = max(b[3] for b in bboxes)

            # 繪製矩形
            cv2.rectangle(
                img,
                (all_left, all_top),
                (all_right, all_bottom),
                (0, 0, 255),  # 紅色框
                2
            )

            # 添加標籤
            label = f"{entity['type']} ({entity['score']:.2f})"
            cv2.putText(
                img, label,
                (all_left, all_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        # 轉換為base64用於API返回
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
