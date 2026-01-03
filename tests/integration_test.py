import os
import time
import pytest

from pathlib import Path

from deid_pipeline import DeidPipeline

if not os.getenv("EDGE_DEID_RUN_INTEGRATION_TESTS"):
    pytest.skip("Integration tests are opt-in. Set EDGE_DEID_RUN_INTEGRATION_TESTS=1.", allow_module_level=True)

@pytest.mark.parametrize("file_path", [
    "samples/sample_contract.pdf",
    "samples/personal_info.docx",
    "samples/medical_record.txt",
])
def test_document_pipeline(file_path: str, tmp_path: Path):
    if not os.path.exists(file_path):
        pytest.skip(f"Sample missing: {file_path}")

    pipeline = DeidPipeline(language="zh")

    t0 = time.perf_counter()
    result = pipeline.process(file_path, output_mode="replace", output_dir=tmp_path)
    wall_s = time.perf_counter() - t0

    assert isinstance(result.text, str)
    assert isinstance(result.entities, list)
    assert isinstance(result.timings_ms, dict)

    output_path = result.artifacts.get("output_path")
    if output_path:
        assert os.path.exists(str(output_path))

    print(f"Processed {file_path} in {wall_s:.2f}s | entities={len(result.entities)}")


def test_image_pipeline(tmp_path: Path):
    img_path = "samples/tw_id_card.jpg"
    if not os.path.exists(img_path):
        pytest.skip("Image sample missing")

    pipeline = DeidPipeline(language="zh")
    result = pipeline.process(img_path, output_mode="replace", output_dir=tmp_path)

    assert isinstance(result.text, str)
    assert isinstance(result.entities, list)
    assert isinstance(result.artifacts, dict)
