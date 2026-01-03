from __future__ import annotations

from pathlib import Path

from deid_pipeline import DeidPipeline


def test_csv_handler_rebuild_writes_output_file(tmp_path: Path):
    input_path = tmp_path / "sample.csv"
    input_path.write_text("id,phone\nA123456789,0912345678\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    result = DeidPipeline(language="zh").process(
        str(input_path),
        output_mode="replace",
        output_dir=str(output_dir),
    )

    assert "output_path" in result.artifacts
    out_path = Path(result.artifacts["output_path"])
    assert out_path.exists()

    content = out_path.read_text(encoding="utf-8", errors="replace")
    assert "A123456789" not in content
    assert "0912345678" not in content
