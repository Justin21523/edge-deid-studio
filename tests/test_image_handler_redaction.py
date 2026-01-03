from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from deid_pipeline.handlers.base import SegmentSpec, build_document_from_segments
from deid_pipeline.handlers.image import ImageHandler


def test_image_handler_redacts_overlapping_segments(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpg"
    img = np.full((80, 120, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(img_path), img)

    segments = [
        SegmentSpec(text="John", page_index=0, bbox=(10, 10, 30, 30)),
        SegmentSpec(text="Smith", page_index=0, bbox=(40, 10, 70, 30)),
    ]
    document = build_document_from_segments(
        input_path=img_path, language="en", segments=segments, separator=" "
    )

    entity = {
        "type": "NAME",
        "span": (0, len(document.text)),
        "text": document.text,
        "confidence": 1.0,
        "source": "test",
        "language": "en",
    }

    handler = ImageHandler()
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
    out_path = Path(str(artifacts["output_path"]))
    assert out_path.exists()

    assert artifacts.get("redaction_count") == 2
    rects = artifacts.get("redaction_rects") or []
    assert {r.get("strategy") for r in rects} == {"segment_overlap"}

    out_img = cv2.imread(str(out_path))
    assert out_img is not None

    # Centers of redacted boxes should be near-black.
    assert int(out_img[20, 20, 0]) < 20
    assert int(out_img[20, 55, 0]) < 20

    # A pixel outside the redacted regions should remain near-white.
    assert int(out_img[5, 5, 0]) > 200
