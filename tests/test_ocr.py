import cv2
from src.deid_pipeline.parser.ocr import OCRAdapter

def test_ocr_engines():
    image_path = "test_data/chinese_document.jpg"
    img = cv2.imread(image_path)

    # 測試Tesseract
    tesseract_ocr = OCRAdapter(engine="tesseract", lang="zh")
    t_text, _ = tesseract_ocr.recognize(img)
    print(f"Tesseract 辨識結果:\n{t_text}\n")

    # 測試EasyOCR
    easyocr_ocr = OCRAdapter(engine="easyocr", lang="zh")
    e_text, _ = easyocr_ocr.recognize(img)
    print(f"EasyOCR 辨識結果:\n{e_text}\n")

    # 測試自動選擇
    auto_ocr = OCRAdapter(engine="auto", lang="zh")
    a_text, _ = auto_ocr.recognize(img)
    print(f"Auto 引擎結果:\n{a_text}\n")
