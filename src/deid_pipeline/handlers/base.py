from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple


BBox = Tuple[int, int, int, int]  # (left, top, right, bottom)
CellRef = Dict[str, Any]


@dataclass(frozen=True)
class TextSegment:
    """A slice of extracted text with optional structural anchors."""

    segment_id: int
    start: int
    end: int
    text: str

    page_index: Optional[int] = None
    bbox: Optional[BBox] = None
    cell: Optional[CellRef] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentSpec:
    """Segment spec used to build an ExtractedDocument."""

    text: str
    page_index: Optional[int] = None
    bbox: Optional[BBox] = None
    cell: Optional[CellRef] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    """A normalized extracted document representation.

    `text` is a concatenation of `segments[*].text` joined by a handler-defined separator.
    """

    input_path: Path
    file_extension: str
    language: str
    text: str
    segments: Sequence[TextSegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def context_hash(self) -> str:
        """A stable hash used for deterministic replacement within this document."""

        digest = hashlib.sha256(self.text.encode("utf-8", errors="replace")).hexdigest()
        return digest


def build_document_from_segments(
    *,
    input_path: Path,
    language: str,
    file_extension: Optional[str] = None,
    segments: Sequence[SegmentSpec],
    separator: str = "\n",
    metadata: Optional[Dict[str, Any]] = None,
) -> ExtractedDocument:
    """Build an ExtractedDocument from a list of segment specs."""

    parts: list[str] = []
    built_segments: list[TextSegment] = []
    cursor = 0

    for idx, spec in enumerate(segments):
        seg_text = spec.text or ""
        start = cursor
        end = start + len(seg_text)
        built_segments.append(
            TextSegment(
                segment_id=idx,
                start=start,
                end=end,
                text=seg_text,
                page_index=spec.page_index,
                bbox=spec.bbox,
                cell=spec.cell,
                metadata=dict(spec.metadata),
            )
        )
        parts.append(seg_text)
        cursor = end
        if idx != len(segments) - 1:
            parts.append(separator)
            cursor += len(separator)

    return ExtractedDocument(
        input_path=input_path,
        file_extension=(file_extension or input_path.suffix.lower()),
        language=language,
        text="".join(parts),
        segments=built_segments,
        metadata=dict(metadata or {}),
    )


class FormatHandler(Protocol):
    """Handler interface for file formats."""

    extensions: Sequence[str]

    def extract(self, input_path: Path, *, language: str) -> ExtractedDocument: ...

    def rebuild(
        self,
        document: ExtractedDocument,
        *,
        output_text: str,
        entities: list[dict],
        replacement_map: Dict[str, str],
        events: list[dict],
        output_dir: Optional[Path] = None,
        mode: str = "replace",
    ) -> Dict[str, Any]: ...
