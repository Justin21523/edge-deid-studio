# examples/advanced_usage.py

from sensitive_data_generator.dataset_generator import MultiFormatDatasetGenerator
from sensitive_data_generator.advanced_file_writers import AdvancedFileWriter
from sensitive_data_generator.advanced_formatters import AdvancedDataFormatter


def demo_full_dataset():
    """Generate a full multi-format demo dataset."""
    dataset = MultiFormatDatasetGenerator.generate_full_dataset(
        output_dir="advanced_test_dataset",
        num_items=100
    )
    print(f"Generated {len(dataset)} items")


def demo_single_documents():
    """Generate a few complex single-document examples."""
    # Contract-like document
    contract = AdvancedDataFormatter.generate_contract_document()
    AdvancedFileWriter.create_complex_pdf(contract, "contracts", "sample_contract.pdf")
    AdvancedFileWriter.create_word_document(contract, "contracts", "sample_contract.docx")
    # Financial spreadsheet
    AdvancedFileWriter.create_excel_spreadsheet("financial", "financial_sample.xlsx")
    # Scanned medical report
    medical_report = AdvancedDataFormatter.generate_medical_report()
    AdvancedFileWriter.create_scanned_document(medical_report, "scanned", "medical_report.png")


if __name__ == "__main__":
    demo_full_dataset()
    demo_single_documents()
