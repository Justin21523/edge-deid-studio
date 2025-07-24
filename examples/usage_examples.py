# examples/usage_examples.py

from sensitive_data_generator.generators import PIIGenerator
from sensitive_data_generator.formatters import DataFormatter
from sensitive_data_generator.file_writers import FileWriter

def basic_usage():
    """基本使用範例 (Basic Usage)"""
    print("台灣身分證:", PIIGenerator.generate_tw_id())
    print("手機號碼:", PIIGenerator.generate_tw_phone())
    print("地址:", PIIGenerator.generate_tw_address())
    print("姓名:", PIIGenerator.generate_tw_name())
    print("病歷號:", PIIGenerator.generate_medical_record())
    pii_type, pii_value = PIIGenerator.generate_random_pii()
    print(f"隨機PII: {pii_type} - {pii_value}")

def document_generation():
    """文件生成範例 (Document Generation)"""
    # 醫療記錄
    medical = DataFormatter.generate_medical_record()
    print(medical)
    FileWriter.write_pdf_file(medical, "output/medical_records", "patient_record.pdf")
    # 財務文件
    finance = DataFormatter.generate_financial_document()
    print(finance)
    FileWriter.write_image_file(finance, "output/financial_docs", "bank_statement.png")

def batch_generation():
    """批量生成測試資料集 (Batch Dataset Generation)"""
    dataset = FileWriter.generate_dataset(
        output_dir="test_dataset",
        num_items=100,
        formats=["txt", "pdf", "image", "csv", "json"]
    )
    print(f"已生成 {len(dataset)} 個測試項目")

if __name__ == "__main__":
    basic_usage()
    document_generation()
    batch_generation()
