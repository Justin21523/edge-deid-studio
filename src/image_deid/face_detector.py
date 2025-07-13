import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )

    def detect(self, image: np.ndarray, min_confidence=0.7) -> list:
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype("int"))
        return faces
