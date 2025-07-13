import cv2
import numpy as np
from paddleocr import PaddleOCR

class OCRProcessor:
    def __init__(self, lang='ch'):
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # 自動對比度調整 (CLAHE)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    def extract_text(self, image: np.ndarray) -> str:
        processed = self.preprocess_image(image)
        result = self.ocr_engine.ocr(processed, cls=True)
        # result: List[List[ (box, (text, conf)) ]]
        lines = []
        for line in result:
            text, _ = line[1]
            lines.append(text)
        return "\n".join(lines)
