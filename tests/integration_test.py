# tests/integration_test.py
import os, time
import pytest

from deid_pipeline.parser.text_extractor import extract_text
from deid_pipeline.pii import get_detector
from deid_pipeline.utils.replacer import Replacer
from deid_pipeline.parser.ocr_processor import OCRPIIProcessor

@pytest.mark.parametrize("file_path", [
    "samples/sample_contract.pdf",
    "samples/personal_info.docx",
    "samples/medical_record.txt",
])
def test_document_pipeline(file_path):
    assert os.path.exists(file_path), f"{file_path} missing"

    # 文字提取
    t0 = time.time()
    text, _ = extract_text(file_path)
    dt = time.time() - t0
    assert len(text) > 0
    print(f"Extracted {len(text)} chars in {dt:.2f}s")

    # PII 檢測
    det = get_detector("zh")
    t1 = time.time()
    entities = det.detect(text)
    ddt = time.time() - t1
    print(f"Detected {len(entities)} entities in {ddt:.2f}s")

    # 替換
    rep = Replacer()
    t2 = time.time()
    clean, events = rep.replace(text, entities)
    rdt = time.time() - t2
    assert clean != text or not entities
    print(f"Replaced in {rdt:.2f}s")

def test_image_pipeline():
    img_path = "samples/tw_id_card.jpg"
    if not os.path.exists(img_path):
        pytest.skip("Image sample missing")
    proc = OCRPIIProcessor(lang="zh")
    result = proc.process_image(img_path)
    assert result["status"] == "success"
    assert "original_text" in result
    assert isinstance(result["entities"], list)
