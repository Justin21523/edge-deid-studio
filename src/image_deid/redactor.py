import cv2
import numpy as np

class Redactor:
    def __init__(self):
        self.method = "pixelate"

    def mask(self, image: np.ndarray, box: tuple) -> np.ndarray:
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        # 像素化
        factor = 8
        small = cv2.resize(roi, ( (x2-x1)//factor, (y2-y1)//factor ), interpolation=cv2.INTER_NEAREST)
        pixelated = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = pixelated
        return image

    def blur(self, image: np.ndarray, box: tuple) -> np.ndarray:
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (23, 23), 30)
        image[y1:y2, x1:x2] = blurred
        return image
