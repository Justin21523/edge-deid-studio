from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict


TextSpan = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # (left, top, right, bottom)


class CellRef(TypedDict, total=False):
    sheet: str
    address: str  # e.g. "A1"
    row: int
    col: int


class Entity(TypedDict, total=False):
    """Canonical entity contract (Phase 1).

    Notes:
    - `confidence` is the canonical field. `score` is kept as a legacy alias.
    - `span` is always in the coordinate space of the extracted text for the given scope
      (document-level concatenation for now).
    """

    # Identity / classification
    type: str
    confidence: float
    score: float  # legacy alias for confidence
    source: str
    language: str

    # Text-level anchor
    span: TextSpan
    text: str  # original substring (best-effort)

    # Document-level anchors (optional; used for rebuild)
    page_index: int
    bbox: BBox
    cell: CellRef

    metadata: Dict[str, Any]


class DeidEvent(TypedDict, total=False):
    """Replacement/masking audit event."""

    entity_type: str
    original: str
    replacement: str
    span: TextSpan
    source: str
    metadata: Dict[str, Any]


@dataclass
class DeidResult:
    """Stable pipeline result contract (Phase 1)."""

    entities: List[Entity]
    text: str
    replacement_map: Dict[str, str] = field(default_factory=dict)
    events: List[DeidEvent] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "text": self.text,
            "entities": list(self.entities),
            "replacement_map": dict(self.replacement_map),
            "events": list(self.events),
            "timings_ms": dict(self.timings_ms),
            "artifacts": dict(self.artifacts),
        }


def normalize_entity(
    entity: Dict[str, Any],
    *,
    language: str,
    source: Optional[str] = None,
    text: Optional[str] = None,
) -> Entity:
    """Normalize a detector entity into the canonical contract."""

    normalized: Entity = {}

    if "type" not in entity:
        raise KeyError("Entity is missing required field: type")

    normalized["type"] = str(entity["type"])

    # Normalize span (list -> tuple, strings -> ints where possible).
    raw_span = entity.get("span")
    if raw_span is not None:
        start, end = raw_span
        normalized["span"] = (int(start), int(end))

    # Confidence/score normalization (keep both for compatibility).
    if "confidence" in entity:
        conf = float(entity["confidence"])
    elif "score" in entity:
        conf = float(entity["score"])
    else:
        conf = 0.0
    normalized["confidence"] = conf
    normalized["score"] = conf

    normalized["source"] = str(source or entity.get("source") or "unknown")
    normalized["language"] = str(entity.get("language") or language)

    if text is not None and "span" in normalized and "text" not in entity:
        s, e = normalized["span"]
        normalized["text"] = text[s:e]
    elif "text" in entity:
        normalized["text"] = str(entity["text"])

    # Optional anchors
    if "page_index" in entity and entity["page_index"] is not None:
        normalized["page_index"] = int(entity["page_index"])
    if "bbox" in entity and entity["bbox"] is not None:
        left, top, right, bottom = entity["bbox"]
        normalized["bbox"] = (int(left), int(top), int(right), int(bottom))
    if "cell" in entity and entity["cell"] is not None:
        normalized["cell"] = dict(entity["cell"])

    if "metadata" in entity and isinstance(entity["metadata"], dict):
        normalized["metadata"] = dict(entity["metadata"])

    return normalized


def replacement_key(entity_type: str, original: str) -> str:
    """Phase-1 replacement map key. Phase-2 will extend this with context hashing."""

    return f"{entity_type}:{original}"
