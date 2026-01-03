import os

import pytest

if not os.getenv("EDGE_DEID_RUN_INTEGRATION_TESTS"):
    pytest.skip("Integration tests are opt-in. Set EDGE_DEID_RUN_INTEGRATION_TESTS=1.", allow_module_level=True)

from deid_pipeline import DeidPipeline  # noqa: E402
from sensitive_data_generator import FileWriter  # noqa: E402


def test_pipeline_runs_on_generated_txt(tmp_path):
    dataset = FileWriter.generate_dataset(
        output_dir=str(tmp_path / "test_data"),
        num_items=3,
        formats=["txt"],
    )

    pipeline = DeidPipeline(language="zh")
    for item in dataset:
        txt_files = [f for f in item["files"] if f["format"] == "txt"]
        assert txt_files, "Expected at least one txt output"
        result = pipeline.process(txt_files[0]["path"])
        assert isinstance(result.text, str)
