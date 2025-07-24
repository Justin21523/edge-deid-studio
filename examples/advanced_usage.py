# examples/advanced_usage.py

import os
from sensitive_data_generator.dataset_generator import MultiFormatDatasetGenerator
from sensitive_data_generator.advanced_file_writers import AdvancedFileWriter
from sensitive_data_generator.advanced_formatters import AdvancedDataFormatter

def demo_full_dataset():
    """生成完整多格式測試資料集 (Full multi-format demo)"""
    dataset = MultiFormatDatasetGenerator.generate_full_dataset(
        output_dir="advanced_test_dataset",
        num_items=100
    )
    print(f"Generated {len(dataset)} items")

def demo_single_documents():
    """單一文件生成範例 (Single document demo)"""
    # 合約
    contract = AdvancedDataFormatter.generate_contract_document()
    AdvancedFileWriter.create_complex_pdf(contract, "contracts", "sample_contract.pdf")
    AdvancedFileWriter.create_word_document(contract, "contracts", "sample_contract.docx")
    # 財務 Excel
    AdvancedFileWriter.create_excel_spreadsheet("financial", "financial_sample.xlsx")
    # 醫療掃描
    medical_report = AdvancedDataFormatter.generate_medical_report()
    AdvancedFileWriter.create_scanned_document(medical_report, "scanned", "medical_report.png")

if __name__ == "__main__":
    demo_full_dataset()
    demo_single_documents()
