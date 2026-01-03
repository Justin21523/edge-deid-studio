from __future__ import annotations

from deid_pipeline.core.anchors import attach_segment_anchors
from deid_pipeline.handlers.base import TextSegment


def test_attach_segment_anchors_adds_bbox_and_page_index():
    segments = [
        TextSegment(
            segment_id=0,
            start=0,
            end=10,
            text="0123456789",
            page_index=2,
            bbox=(1, 2, 3, 4),
        )
    ]
    entities = [{"type": "ID", "span": (2, 5), "score": 1.0, "source": "test"}]

    attach_segment_anchors(entities, segments)

    assert entities[0]["page_index"] == 2
    assert entities[0]["bbox"] == (1, 2, 3, 4)


def test_attach_segment_anchors_skips_cross_segment_spans():
    segments = [
        TextSegment(segment_id=0, start=0, end=3, text="abc", page_index=0, bbox=(0, 0, 1, 1)),
        TextSegment(segment_id=1, start=4, end=7, text="def", page_index=0, bbox=(1, 0, 2, 1)),
    ]
    # Span crosses the gap between segments (3..4).
    entities = [{"type": "NAME", "span": (2, 6), "score": 1.0, "source": "test"}]

    attach_segment_anchors(entities, segments)

    assert "bbox" not in entities[0]
    assert "page_index" not in entities[0]
