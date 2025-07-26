# src/deid_pipeline/parser/ocr.py
import easyocr
from ..config import Config

_OCR_READER = None

def get_ocr_reader(langs: list[str] | None = None):
    """Singleton EasyOCR reader, default to Config.OCR_LANGUAGES."""
    global _OCR_READER
    if _OCR_READER is None:
        languages = langs or Config.OCR_LANGUAGES
        _OCR_READER = easyocr.Reader(languages, gpu=False)
    return _OCR_READER
