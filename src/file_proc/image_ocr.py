import cv2
import numpy as np
from paddleocr import PaddleOCR

class OCRProcessor:
    def __init__(self, lang='ch'):
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # TODO: 影像前處理
        return image

    def extract_text(self, image: np.ndarray) -> str:
        # TODO: 呼叫 ocr_engine
        return ""
