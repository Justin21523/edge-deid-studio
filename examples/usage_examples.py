# examples/usage_examples.py

from sensitive_data_generator.generators import PIIGenerator
from sensitive_data_generator.formatters import DataFormatter
from sensitive_data_generator.file_writers import FileWriter


def basic_usage():
    """Basic usage examples."""
    print("Taiwan ID:", PIIGenerator.generate_tw_id())
    print("Phone number:", PIIGenerator.generate_tw_phone())
    print("Address:", PIIGenerator.generate_tw_address())
    print("Name:", PIIGenerator.generate_tw_name())
    print("Medical record ID:", PIIGenerator.generate_medical_record())
    pii_type, pii_value = PIIGenerator.generate_random_pii()
    print(f"Random PII: {pii_type} - {pii_value}")


def document_generation():
    """Document generation examples."""
    # Medical record
    medical = DataFormatter.generate_medical_record()
    print(medical)
    FileWriter.write_pdf_file(medical, "output/medical_records", "patient_record.pdf")
    # Financial document
    finance = DataFormatter.generate_financial_document()
    print(finance)
    FileWriter.write_image_file(finance, "output/financial_docs", "bank_statement.png")


def batch_generation():
    """Generate a batch dataset for local testing."""
    dataset = FileWriter.generate_dataset(
        output_dir="test_dataset",
        num_items=100,
        formats=["txt", "pdf", "image", "csv", "json"]
    )
    print(f"Generated {len(dataset)} test items")


if __name__ == "__main__":
    basic_usage()
    document_generation()
    batch_generation()
