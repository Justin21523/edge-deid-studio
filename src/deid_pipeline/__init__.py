# src/deid_pipeline/__init__.py
from deid_pipeline.parser.text_extractor import extract_text
from deid_pipeline.image_deid.processor import ImageDeidProcessor
from deid_pipeline.pii.detectors import get_detector
from deid_pipeline.pii.utils.replacer import Replacer

class DeidResult:
    """
    把偵測結果包成 result.entities 及 result.text
    供原始測試呼叫。
    """
    def __init__(self, entities, text):
        self.entities = entities
        self.text = text

class DeidPipeline:
    """
    統一介面：.process(input_path, output_mode, generate_report)
    不同副檔用 extract_text 或 OCRPIIProcessor。
    """
    def __init__(self, language: str = "zh"):
        self.lang = language
        self.detector = get_detector(language)
        self.replacer = Replacer()
        self.ocr_proc = ImageDeidProcessor(lang=language)

    def process(self, input_path: str, output_mode: str = "replacement", generate_report: bool = False):
        # 1. 讀文字或影像
        suffix = input_path.lower().split(".")[-1]
        if suffix in ("txt", "docx", "pdf"):
            text, _ = extract_text(input_path)
        else:
            ocr_res = self.ocr_proc.process_image(input_path)
            text = ocr_res.get("original_text", "")

        # 2. 偵測
        entities = self.detector.detect(text)

        # 3. 替換
        clean_text, _ = self.replacer.replace(text, entities)

        # 4. 回傳
        return DeidResult(entities=entities, text=clean_text)
