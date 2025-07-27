# src/deid_pipeline/parser/ocr.py
import pytesseract
from PIL import Image
import cv2
import numpy as np
import easyocr
from ..config import Config
import logging

logger = logging.getLogger(__name__)

class OCRAdapter:
    """統一的OCR介面，支援Tesseract和EasyOCR"""
    def __init__(self, engine="auto", lang="zh"):
        self.engine = engine
        self.lang = lang
        self.tesseract_config = self._get_tesseract_config()

        if engine == "easyocr" or engine == "auto":
            try:
                self.easyocr_reader = easyocr.Reader(
                    ['ch_tra' if lang == "zh" else 'en'],
                    gpu=Config.USE_GPU
                )
                self.active_engine = "easyocr"
                logger.info("EasyOCR引擎初始化成功")
            except Exception as e:
                logger.warning(f"EasyOCR初始化失敗: {str(e)}")
                self.active_engine = "tesseract"
        else:
            self.active_engine = "tesseract"

        if self.active_engine == "tesseract":
            logger.info(f"使用Tesseract引擎，語言: {lang}")

    def _get_tesseract_config(self):
        """繁體中文專用配置"""
        if self.lang == "zh":
            return r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        return r'--oem 3 --psm 6'

    def _preprocess_image(self, image):
        """影像預處理增強OCR準確度"""
        # 轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 自適應閾值二值化
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 中值濾波去噪
        processed = cv2.medianBlur(processed, 3)
        return processed

    def recognize(self, image):
        """辨識影像中的文字，返回文字和文字框"""
        if isinstance(image, str):  # 檔案路徑
            image = cv2.imread(image)

        processed_img = self._preprocess_image(image)

        if self.active_engine == "easyocr":
            return self._recognize_easyocr(processed_img)
        else:
            return self._recognize_tesseract(processed_img)

    def _recognize_tesseract(self, image):
        """Tesseract辨識實現"""
        # 轉換為PIL格式
        pil_img = Image.fromarray(image)

        # 繁體中文專用配置
        lang_code = "chi_tra" if self.lang == "zh" else "eng"

        data = pytesseract.image_to_data(
            pil_img,
            lang=lang_code,
            output_type=pytesseract.Output.DICT,
            config=self.tesseract_config
        )

        text_blocks = []
        full_text = ""
        current_line = []
        prev_top = -1

        for i in range(len(data['text'])):
            conf = int(data['conf'][i])
            text = data['text'][i].strip()

            if conf > Config.OCR_CONFIDENCE_THRESHOLD and text:
                left, top, width, height = (
                    data['left'][i], data['top'][i],
                    data['width'][i], data['height'][i]
                )

                # 換行檢測
                if prev_top != -1 and abs(top - prev_top) > height * 0.5:
                    line_text = " ".join(current_line)
                    full_text += line_text + "\n"
                    current_line = []

                current_line.append(text)
                prev_top = top

                text_blocks.append({
                    'text': text,
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height,
                    'conf': conf
                })

        # 添加最後一行
        if current_line:
            full_text += " ".join(current_line)

        return full_text.strip(), text_blocks

    def _recognize_easyocr(self, image):
        """EasyOCR辨識實現"""
        results = self.easyocr_reader.readtext(
            image,
            detail=1,
            paragraph=False
        )

        text_blocks = []
        full_text = ""
        current_line = []
        prev_top = -1

        for result in results:
            bbox, text, conf = result
            if conf < Config.OCR_CONFIDENCE_THRESHOLD:
                continue

            # 提取邊界框座標
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            left, top = top_left
            width = bottom_right[0] - left
            height = bottom_right[1] - top

            # 換行檢測
            if prev_top != -1 and abs(top - prev_top) > height * 0.5:
                line_text = " ".join(current_line)
                full_text += line_text + "\n"
                current_line = []

            current_line.append(text)
            prev_top = top

            text_blocks.append({
                'text': text,
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'conf': conf
            })

        # 添加最後一行
        if current_line:
            full_text += " ".join(current_line)

        return full_text.strip(), text_blocks

# 單例存取點
_ocr_instance = None

def get_ocr_reader(engine="auto", lang="zh"):
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = OCRAdapter(engine, lang)
    return _ocr_instance
