from __future__ import annotations

from pathlib import Path

import pytest

from deid_pipeline.handlers.pdf import PdfHandler


def _make_searchable_pdf(path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "ID: A123456789", fontsize=12)
    page.insert_text((72, 92), "Name: Alice", fontsize=12)
    doc.save(str(path))
    doc.close()


def test_pdf_redaction_uses_text_search_when_available(tmp_path: Path) -> None:
    pytest.importorskip("fitz")

    pdf_path = tmp_path / "sample.pdf"
    _make_searchable_pdf(pdf_path)

    handler = PdfHandler()
    document = handler.extract(pdf_path, language="en")

    target_segment = next(
        (segment for segment in document.segments if "A123456789" in segment.text),
        None,
    )
    assert target_segment is not None
    assert target_segment.page_index == 0
    assert target_segment.bbox is not None

    start = document.text.find("A123456789")
    assert start >= 0

    entity = {
        "type": "TW_ID",
        "span": (start, start + len("A123456789")),
        "text": "A123456789",
        "page_index": target_segment.page_index,
        "bbox": target_segment.bbox,
        "confidence": 1.0,
        "source": "test",
        "language": "en",
    }

    artifacts = handler.rebuild(
        document,
        output_text=document.text,
        entities=[entity],
        replacement_map={},
        events=[],
        output_dir=tmp_path,
        mode="replace",
    )

    assert artifacts.get("rebuild_supported") is True
    assert artifacts.get("output_path")
    assert Path(str(artifacts["output_path"])).exists()

    rects = artifacts.get("redaction_rects") or []
    assert rects

    bbox = target_segment.bbox
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    assert any(
        (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]) < bbox_area for r in rects
    )
