# src/deid_pipeline/parser/ocr.py
import easyocr
_OCR_READER = None

def get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        _OCR_READER = easyocr.Reader(["en","ch_sim"], gpu=False)
    return _OCR_READER
