import cv2
import numpy as np
from .text_detection import TextDetector
from .face_detector import FaceDetector
from .plate_detector import PlateDetector
from .redactor import Redactor

class ImagePIIPipeline:
    def __init__(self, onnx_session):
        self.text_detector = TextDetector()
        self.face_detector = FaceDetector()
        self.plate_detector = PlateDetector()
        self.ner_session = onnx_session
        self.redactor = Redactor()

    def detect(self, image: np.ndarray) -> dict:
        results = {
            'text_boxes': self.text_detector.detect(image),
            'faces': self.face_detector.detect(image),
            'plates': self.plate_detector.detect(image)
        }
        return results

    def redact_image(self, image: np.ndarray, results: dict) -> np.ndarray:
        # 文字遮蔽
        for box in results['text_boxes']:
            x1, y1, x2, y2 = box
            roi = image[y1:y2, x1:x2]
            text = self._recognize_text(roi)
            if self._is_sensitive(text):
                image = self.redactor.mask(image, box)
        # 視覺實體遮蔽
        for entity in results['faces'] + results['plates']:
            image = self.redactor.blur(image, tuple(entity))
        return image

    def _recognize_text(self, roi: np.ndarray) -> str:
        # TODO: run ONNX text recognition
        return ""

    def _is_sensitive(self, text: str) -> bool:
        # TODO: run ONNX NER inference
        return False
