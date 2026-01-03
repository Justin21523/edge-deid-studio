from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np

from .ocr import get_ocr_reader
from ..config import Config
from ..pii.utils import logger


BBox = Tuple[int, int, int, int, int]  # (page_index, left, top, right, bottom)
OffsetMapEntry = Tuple[BBox, int]  # (bbox, char_index)


def extract_text(file_path: str, lang: str = "zh", ocr_engine: str = "auto") -> tuple[str, List[OffsetMapEntry]]:
    """Convenience wrapper used by CLI/tests."""

    return TextExtractor(lang=lang, ocr_engine=ocr_engine).extract_text(file_path)


class TextExtractor:
    """Unified text extraction supporting multiple file formats.

    This module avoids importing optional heavy dependencies (PyMuPDF, python-docx,
    OCR libraries) at import-time. If a dependency is missing, extraction returns
    an empty result and logs an error.
    """

    def __init__(self, lang: str = "zh", ocr_engine: str = "auto"):
        self.lang = lang
        self.ocr_engine = ocr_engine
        self.ocr_processor = None

    def init_ocr(self) -> None:
        if self.ocr_processor is not None:
            return
        try:
            self.ocr_processor = get_ocr_reader(self.ocr_engine, self.lang)
        except Exception as exc:
            logger.warning("OCR is unavailable (%s). OCR fallback will be disabled.", exc)
            self.ocr_processor = None

    def extract_text(self, file_path: str) -> tuple[str, List[OffsetMapEntry]]:
        """Extract text and return (text, offset_map)."""

        start_time = time.perf_counter()
        ext = Path(file_path).suffix.lower()

        try:
            if ext == ".txt":
                return self._extract_txt(file_path)
            if ext == ".docx":
                return self._extract_docx(file_path)
            if ext == ".html":
                return self._extract_html(file_path)
            if ext == ".pdf":
                return self._extract_pdf(file_path)
            if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
                return self._extract_image(file_path)

            raise ValueError(f"Unsupported file extension: {ext}")
        except Exception as exc:
            logger.error("Text extraction failed for %s: %s", file_path, str(exc))
            return "", []
        finally:
            elapsed = time.perf_counter() - start_time
            logger.info("Text extraction completed: %s (%.2fs)", file_path, elapsed)

    def _unknown_bbox_map(self, text: str) -> List[OffsetMapEntry]:
        unknown_bbox: BBox = (-1, -1, -1, -1, -1)
        return [(unknown_bbox, i) for i in range(len(text))]

    def _extract_txt(self, file_path: str) -> tuple[str, List[OffsetMapEntry]]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return text, self._unknown_bbox_map(text)

    def _extract_docx(self, file_path: str) -> tuple[str, List[OffsetMapEntry]]:
        try:
            from docx import Document  # type: ignore
        except Exception as exc:
            raise ImportError("python-docx is required to extract .docx files") from exc

        doc = Document(file_path)
        parts: List[str] = []
        for para in doc.paragraphs:
            parts.append(para.text)
        text = "\n".join(parts) + ("\n" if parts else "")
        return text, self._unknown_bbox_map(text)

    def _extract_html(self, file_path: str) -> tuple[str, List[OffsetMapEntry]]:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except Exception as exc:
            raise ImportError("beautifulsoup4 is required to extract .html files") from exc

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\s+", " ", text).strip()
        return text, self._unknown_bbox_map(text)

    def _extract_pdf(self, file_path: str) -> tuple[str, List[OffsetMapEntry]]:
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise ImportError("PyMuPDF is required to extract .pdf files") from exc

        self.init_ocr()
        cfg = Config()

        doc = fitz.open(file_path)
        full_text_parts: List[str] = []
        offset_map: List[OffsetMapEntry] = []
        char_index = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text_parts: List[str] = []

            # Extract text blocks first.
            try:
                text_blocks = page.get_text("blocks", sort=True)
            except Exception as exc:
                logger.warning("Failed to extract PDF blocks on page %d: %s", page_num, exc)
                text_blocks = []

            for block in text_blocks:
                # PyMuPDF block: (x0, y0, x1, y1, "text", block_no, block_type)
                if len(block) < 7:
                    continue
                if block[6] != 0:
                    continue
                block_text = str(block[4]).strip()
                if not block_text:
                    continue

                page_text_parts.append(block_text)
                left, top, right, bottom = int(block[0]), int(block[1]), int(block[2]), int(block[3])
                bbox: BBox = (page_num, left, top, right, bottom)
                for i in range(len(block_text)):
                    offset_map.append((bbox, char_index + i))
                char_index += len(block_text) + 1  # + newline

            page_text = "\n".join(page_text_parts).strip()

            # OCR fallback for scanned/empty pages.
            if cfg.OCR_ENABLED and len(page_text) < cfg.OCR_THRESHOLD and self.ocr_processor is not None:
                logger.info("OCR fallback triggered for page %d", page_num)
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                # Convert to BGR
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                ocr_text, ocr_blocks = self.ocr_processor.recognize(img)
                page_text = (ocr_text or "").strip()
                for block in ocr_blocks:
                    block_text = str(block.get("text", ""))
                    if not block_text:
                        continue
                    left = int(block["left"])
                    top = int(block["top"])
                    right = int(block["left"] + block["width"])
                    bottom = int(block["top"] + block["height"])
                    bbox = (page_num, left, top, right, bottom)
                    for i in range(len(block_text)):
                        offset_map.append((bbox, char_index + i))
                    char_index += len(block_text) + 1

            if page_text:
                full_text_parts.append(page_text)

        return "\n".join(full_text_parts), offset_map

    def _extract_image(self, file_path: str) -> tuple[str, List[OffsetMapEntry]]:
        self.init_ocr()
        if self.ocr_processor is None:
            raise ImportError("OCR dependencies are not installed; cannot extract from images.")

        text, blocks = self.ocr_processor.recognize(file_path)
        offset_map: List[OffsetMapEntry] = []
        char_index = 0

        for block in blocks:
            block_text = str(block.get("text", ""))
            if not block_text:
                continue

            left = int(block["left"])
            top = int(block["top"])
            right = int(block["left"] + block["width"])
            bottom = int(block["top"] + block["height"])
            bbox: BBox = (0, left, top, right, bottom)

            for i in range(len(block_text)):
                offset_map.append((bbox, char_index + i))
            char_index += len(block_text) + 1

        return (text or "").strip(), offset_map

