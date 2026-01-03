from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

from ..handlers.base import TextSegment
from .contracts import Entity


def attach_segment_anchors(entities: list[Entity], segments: Sequence[TextSegment]) -> None:
    """Attach structural anchors (page/bbox/cell) to entities based on segment spans.

    This is a best-effort mapping:
    - An anchor is only attached when an entity span is fully contained in a single segment.
    - If a span crosses segment boundaries, no anchors are attached.
    """

    if not entities or not segments:
        return

    ordered_segments = sorted(segments, key=lambda s: s.start)
    starts = [seg.start for seg in ordered_segments]

    for entity in entities:
        span = entity.get("span")
        if not span:
            continue

        start, end = span
        if start is None or end is None:
            continue

        idx = bisect_right(starts, int(start)) - 1
        if idx < 0:
            continue

        segment = ordered_segments[idx]
        if not (segment.start <= start < segment.end):
            continue
        if end > segment.end:
            continue

        if segment.page_index is not None:
            entity["page_index"] = int(segment.page_index)
        if segment.bbox is not None:
            entity["bbox"] = tuple(segment.bbox)  # type: ignore[assignment]
        if segment.cell is not None:
            entity["cell"] = dict(segment.cell)
