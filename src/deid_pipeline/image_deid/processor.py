# src/deid_pipeline/image_deid/processor.py
from __future__ import annotations

import os
import time
from typing import Dict

import cv2
import numpy as np

from ..parser.text_extractor import TextExtractor
from ..pii.detectors import get_detector
from ..pii.utils.replacer import Replacer

class ImageDeidProcessor:
    def __init__(self, lang: str = "zh", ocr_engine: str = "auto"):
        self.extractor = TextExtractor(lang=lang, ocr_engine=ocr_engine)
        self.detector = get_detector(lang)
        self.replacer = Replacer()
        self.lang = lang

    def process(self, file_path: str, mode: str = "replace") -> Dict:
        start = time.perf_counter()

        t0 = time.perf_counter()
        original_text, offset_map = self.extractor.extract_text(file_path)
        extract_ms = (time.perf_counter() - t0) * 1000.0

        if not original_text.strip():
            return {
                "status": "error",
                "message": "Failed to extract any text from the image.",
                "extract_ms": extract_ms,
                "processing_time": time.perf_counter() - start
            }

        t1 = time.perf_counter()
        entities = self.detector.detect(original_text)
        detect_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        clean_text, events = self.replacer.replace(original_text, entities, mode=mode)
        replace_ms = (time.perf_counter() - t2) * 1000.0

        visual_result = None
        if os.path.splitext(file_path)[1].lower() in ['.jpg', '.jpeg', '.png']:
            # Attach a best-effort bbox to each entity for downstream consumers.
            self._attach_bboxes(entities, offset_map)
            visual_result = self._generate_visual_result(file_path, entities, offset_map)

        return {
            "status": "success",
            "original_text": original_text,
            "clean_text": clean_text,
            "entities": entities,
            "events": events,
            "visual_result": visual_result,
            "extract_ms": extract_ms,
            "detect_ms": detect_ms,
            "replace_ms": replace_ms,
            "processing_time": time.perf_counter() - start
        }

    def process_image(self, file_path: str) -> Dict:
        """Backward-compatible alias for `process()`."""

        return self.process(file_path)

    def _generate_visual_result(self, image_path, entities, offset_map):
        """Generate a JPEG preview with PII bounding boxes overlaid."""

        img = cv2.imread(image_path)
        if img is None:
            return None

        position_index = {pos[1]: pos[0] for pos in offset_map}

        for entity in entities:
            start, end = entity["span"]

            bboxes = []
            for i in range(start, end):
                if i in position_index:
                    page, left, top, right, bottom = position_index[i]
                    if page == 0:  # single image
                        bboxes.append((left, top, right, bottom))

            if not bboxes:
                continue

            all_left = min(b[0] for b in bboxes)
            all_top = min(b[1] for b in bboxes)
            all_right = max(b[2] for b in bboxes)
            all_bottom = max(b[3] for b in bboxes)

            cv2.rectangle(
                img,
                (all_left, all_top),
                (all_right, all_bottom),
                (0, 0, 255),
                2
            )

            label = f"{entity['type']} ({entity['score']:.2f})"
            cv2.putText(
                img, label,
                (all_left, all_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()

    @staticmethod
    def _attach_bboxes(entities, offset_map) -> None:
        """Attach a merged bbox to each entity in-place when an offset map is available."""

        position_index = {pos[1]: pos[0] for pos in offset_map}
        for entity in entities:
            if "span" not in entity:
                continue
            start, end = entity["span"]

            bboxes = []
            for i in range(int(start), int(end)):
                bbox = position_index.get(i)
                if not bbox:
                    continue
                page, left, top, right, bottom = bbox
                if page != 0:
                    continue
                bboxes.append((left, top, right, bottom))

            if not bboxes:
                continue

            entity["page_index"] = 0
            entity["bbox"] = (
                int(min(b[0] for b in bboxes)),
                int(min(b[1] for b in bboxes)),
                int(max(b[2] for b in bboxes)),
                int(max(b[3] for b in bboxes)),
            )
