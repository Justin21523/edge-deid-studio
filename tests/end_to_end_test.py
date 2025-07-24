import os
import time
from deid_pipeline import DeidPipeline
from tests.test_data_factory import TestDataFactory    # 注意路徑


def run_full_pipeline_test(file_type='text'):
    """執行完整流程測試"""
    pipeline = DeidPipeline(language="zh")
    data_factory = TestDataFactory()

    # 根據文件類型生成測試文件
    if file_type == 'text':
        content, pii_data = data_factory.generate_test_document(pii_count=15)
        os.makedirs("tests/test_input", exist_ok=True)
        with open("tests/test_input/sample.txt", "w", encoding="utf-8") as f:
            f.write(content)
        input_path = "test_input/sample.txt"

    elif file_type == 'pdf':
        # 實際專案中需使用PDF生成庫
        input_path = "test_input/medical_report.pdf"

    elif file_type == 'image':
        input_path = "test_input/prescription.jpg"

    # 執行處理流程
    start_time = time.time()
    result = pipeline.process(
        input_path=input_path,
        output_mode="replacement",
        generate_report=True
    )
    elapsed = time.time() - start_time

    # 驗證結果
    verification = {
        "file_type": file_type,
        "pii_count": len(result.entities),
        "processing_time": elapsed,
        "replacement_consistency": True,
        "content_integrity": True
    }

    # 檢查替換一致性
    original_map = {}
    for entity in result.entities:
        if entity['text'] not in original_map:
            original_map[entity['text']] = entity['replaced_with']
        else:
            if original_map[entity['text']] != entity['replaced_with']:
                verification['replacement_consistency'] = False

    # 檢查內容完整性（簡單版本）
    if file_type == 'text':
        with open(input_path, encoding="utf-8") as f:
            original_content = f.read()
        verification['content_integrity'] = len(result.text) > 0.8 * len(original_content)

    return verification

def test_all_formats():
    """測試所有文件格式"""
    formats = ['text', 'pdf', 'image']
    results = {}

    for fmt in formats:
        print(f"測試 {fmt} 文件處理...")
        results[fmt] = run_full_pipeline_test(fmt)

    # 生成測試報告
    print("\n測試結果摘要:")
    for fmt, data in results.items():
        print(f"格式: {fmt.upper()}")
        print(f"  PII偵測數量: {data['pii_count']}")
        print(f"  處理時間: {data['processing_time']:.2f}秒")
        print(f"  替換一致性: {'通過' if data['replacement_consistency'] else '失敗'}")
        print(f"  內容完整性: {'通過' if data['content_integrity'] else '失敗'}")
        print("-" * 40)
