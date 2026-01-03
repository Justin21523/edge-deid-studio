import os
import time

import pytest

from deid_pipeline import DeidPipeline
from tests.test_data_factory import TestDataFactory

if not os.getenv("EDGE_DEID_RUN_E2E_TESTS"):
    pytest.skip("End-to-end tests are opt-in. Set EDGE_DEID_RUN_E2E_TESTS=1.", allow_module_level=True)


def run_full_pipeline_test(file_type='text'):
    """Run an end-to-end pipeline test (opt-in)."""

    pipeline = DeidPipeline(language="zh")
    data_factory = TestDataFactory()

    # Generate test input by file type
    if file_type == 'text':
        content, inserted = data_factory.generate_test_document(pii_count=15)
        os.makedirs("test_input", exist_ok=True)
        with open("test_input/sample.txt", "w", encoding="utf-8") as f:
            f.write(content)
        input_path = "test_input/sample.txt"

    elif file_type == 'pdf':
        # Real projects should generate PDFs via a library (e.g., reportlab)
        input_path = "test_input/medical_report.pdf"

    elif file_type == 'image':
        input_path = "test_input/prescription.jpg"

    # Run pipeline
    start_time = time.time()
    result = pipeline.process(
        input_path=input_path,
        output_mode="replacement",
        generate_report=True
    )
    elapsed = time.time() - start_time

    # Verify results
    verification = {
        "file_type": file_type,
        "pii_count": len(result.entities),
        "processing_time": elapsed,
        "replacement_consistency": True,
        "content_integrity": True
    }

    # Check replacement consistency (same original -> same replacement).
    original_map = {}
    for event in result.events:
        key = (event.get("entity_type"), event.get("original"))
        replacement = event.get("replacement")
        if key not in original_map:
            original_map[key] = replacement
        elif original_map[key] != replacement:
            verification["replacement_consistency"] = False

    # Check content integrity (simple heuristic)
    if file_type == 'text':
        with open(input_path, encoding="utf-8") as f:
            original_content = f.read()
        verification['content_integrity'] = len(result.text) > 0.8 * len(original_content)

    return verification

def test_all_formats():
    """Test multiple formats (opt-in)."""

    formats = ['text', 'pdf', 'image']
    results = {}

    for fmt in formats:
        print(f"Testing format: {fmt} ...")
        results[fmt] = run_full_pipeline_test(fmt)

    # Print report
    print("\nTest summary:")
    for fmt, data in results.items():
        print(f"Format: {fmt.upper()}")
        print(f"  PII entities: {data['pii_count']}")
        print(f"  Time: {data['processing_time']:.2f}s")
        print(f"  Replacement consistency: {'PASS' if data['replacement_consistency'] else 'FAIL'}")
        print(f"  Content integrity: {'PASS' if data['content_integrity'] else 'FAIL'}")
        print("-" * 40)
