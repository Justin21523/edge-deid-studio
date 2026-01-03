from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRTextBlock:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float


class OCRAdapter:
    """A small adapter that unifies Tesseract and EasyOCR under a single interface.

    This module intentionally avoids importing heavy optional dependencies at import-time.
    If an engine is not installed, initialization falls back (or raises) at runtime.
    """

    def __init__(self, engine: str = "auto", lang: str = "zh"):
        self.engine = engine
        self.lang = lang
        self.cfg = Config()

        self.active_engine = "tesseract"
        self.easyocr_reader = None

        if engine in {"easyocr", "auto"}:
            try:
                import easyocr  # type: ignore

                self.easyocr_reader = easyocr.Reader(
                    ["ch_tra" if lang == "zh" else "en"],
                    gpu=self.cfg.USE_GPU,
                )
                self.active_engine = "easyocr"
                logger.info("EasyOCR initialized (lang=%s, gpu=%s)", lang, self.cfg.USE_GPU)
            except Exception as exc:
                logger.warning("EasyOCR unavailable; falling back to Tesseract: %s", exc)
                self.active_engine = "tesseract"

        if self.active_engine == "tesseract":
            # Validate dependency presence early to produce a clearer error.
            try:
                import pytesseract  # noqa: F401
            except Exception as exc:
                raise ImportError(
                    "pytesseract is required for OCR_ENGINE=tesseract (or auto fallback)."
                ) from exc

    def recognize(self, image: str | np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
        """Recognize text and return (full_text, blocks)."""

        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to read image: {image}")
        else:
            img = image

        processed = self._preprocess_image(img)

        if self.active_engine == "easyocr":
            return self._recognize_easyocr(processed)
        return self._recognize_tesseract(processed)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        return cv2.medianBlur(processed, 3)

    def _confidence_threshold(self) -> float:
        """Return the engine-specific confidence threshold."""

        threshold = float(self.cfg.OCR_CONFIDENCE_THRESHOLD)

        # EasyOCR confidence is [0, 1]; Tesseract uses [0, 100] integers.
        if self.active_engine == "easyocr" and threshold > 1:
            return threshold / 100.0
        if self.active_engine == "tesseract" and threshold <= 1:
            return threshold * 100.0
        return threshold

    def _recognize_tesseract(self, image: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        lang_code = "chi_tra" if self.lang == "zh" else "eng"
        config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1" if self.lang == "zh" else r"--oem 3 --psm 6"

        pil_img = Image.fromarray(image)
        data = pytesseract.image_to_data(
            pil_img,
            lang=lang_code,
            output_type=pytesseract.Output.DICT,
            config=config,
        )

        threshold = self._confidence_threshold()
        blocks: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        for i in range(len(data.get("text", []))):
            raw_text = str(data["text"][i]).strip()
            if not raw_text:
                continue

            try:
                conf = float(data["conf"][i])
            except Exception:
                continue

            if conf < threshold:
                continue

            left = int(data["left"][i])
            top = int(data["top"][i])
            width = int(data["width"][i])
            height = int(data["height"][i])

            blocks.append(
                {
                    "text": raw_text,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "conf": conf,
                }
            )
            full_text_parts.append(raw_text)

        return " ".join(full_text_parts).strip(), blocks

    def _recognize_easyocr(self, image: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
        if self.easyocr_reader is None:
            raise RuntimeError("EasyOCR reader is not initialized.")

        threshold = self._confidence_threshold()
        results = self.easyocr_reader.readtext(image, detail=1, paragraph=False)

        blocks: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        for bbox, text, conf in results:
            if float(conf) < threshold:
                continue
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            left, top = top_left
            width = bottom_right[0] - left
            height = bottom_right[1] - top

            blocks.append(
                {
                    "text": text,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "conf": float(conf),
                }
            )
            full_text_parts.append(text)

        return " ".join(full_text_parts).strip(), blocks


_ocr_instances: dict[tuple[str, str], OCRAdapter] = {}


def get_ocr_reader(engine: str = "auto", lang: str = "zh") -> OCRAdapter:
    """Return a cached OCR adapter keyed by (engine, lang)."""

    key = (engine, lang)
    if key not in _ocr_instances:
        _ocr_instances[key] = OCRAdapter(engine=engine, lang=lang)
    return _ocr_instances[key]

