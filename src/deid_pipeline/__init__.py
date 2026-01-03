from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.anchors import attach_segment_anchors
from .core.contracts import DeidResult, Entity, DeidEvent, normalize_entity, replacement_key


class DeidPipeline:
    """Thin orchestration layer for the de-identification pipeline.

    Current behavior (baseline):
    - Extract text (PDF/TXT/DOCX/HTML) or OCR (images)
    - Detect PII entities
    - Replace entities in extracted text

    The format-specific rebuild step will be implemented via pluggable handlers
    in a later refactor phase.
    """

    def __init__(self, language: str = "zh"):
        self.lang = language
        self._detector = None
        self._replacer = None
        self._handler_registry = None

    def _get_detector(self):
        if self._detector is None:
            from .pii.detectors import get_detector

            self._detector = get_detector(self.lang)
        return self._detector

    def _get_replacer(self):
        if self._replacer is None:
            from .pii.utils.replacer import Replacer

            self._replacer = Replacer()
        return self._replacer

    def _get_handler_registry(self):
        if self._handler_registry is None:
            from .handlers.registry import DEFAULT_REGISTRY

            self._handler_registry = DEFAULT_REGISTRY
        return self._handler_registry

    def process(
        self,
        input_path: str,
        output_mode: str = "replacement",
        generate_report: bool = False,  # kept for backward compatibility
        output_dir: str | Path | None = None,
    ) -> DeidResult:
        import time

        path = Path(input_path)
        replace_mode = self._normalize_output_mode(output_mode)

        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

        timings_ms: Dict[str, float] = {}
        artifacts: Dict[str, Any] = {"input_path": str(path)}
        output_dir_path = Path(output_dir) if output_dir is not None else None

        t0 = time.perf_counter()
        handler = self._get_handler_registry().get(path)
        document = handler.extract(path, language=self.lang)
        text = document.text
        timings_ms["extract"] = (time.perf_counter() - t0) * 1000.0
        artifacts["file_extension"] = document.file_extension
        artifacts["document_context_hash"] = document.context_hash

        raw_entities: List[Dict[str, Any]]
        raw_events: List[Dict[str, Any]]
        clean_text: str

        t1 = time.perf_counter()
        detector = self._get_detector()
        raw_entities = detector.detect(text)
        timings_ms["detect"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        replacer = self._get_replacer()
        clean_text, raw_events = replacer.replace(
            text, raw_entities, mode=replace_mode, context_hash=document.context_hash
        )
        timings_ms["replace"] = (time.perf_counter() - t2) * 1000.0

        entities: List[Entity] = [
            normalize_entity(e, language=self.lang, text=text) for e in raw_entities
        ]
        attach_segment_anchors(entities, document.segments)

        events: List[DeidEvent] = []
        replacement_map: Dict[str, str] = {}

        for ev in raw_events:
            original = ev.get("original")
            replacement = ev.get("replacement") or ev.get("fake")
            entity_type = ev.get("type") or ev.get("entity_type")

            if original is not None and replacement is not None and entity_type is not None:
                replacement_map[replacement_key(str(entity_type), str(original))] = str(
                    replacement
                )
                events.append(
                    {
                        "entity_type": str(entity_type),
                        "original": str(original),
                        "replacement": str(replacement),
                        "span": tuple(ev.get("span", (0, 0))),
                        "source": str(ev.get("source", "unknown")),
                    }
                )

        t3 = time.perf_counter()
        artifacts.update(
            handler.rebuild(
                document,
                output_text=clean_text,
                entities=entities,
                replacement_map=replacement_map,
                events=events,
                output_dir=output_dir_path,
                mode=replace_mode,
            )
        )
        timings_ms["rebuild"] = (time.perf_counter() - t3) * 1000.0

        timings_ms["total"] = (
            timings_ms.get("extract", 0.0)
            + timings_ms.get("detect", 0.0)
            + timings_ms.get("replace", 0.0)
            + timings_ms.get("rebuild", 0.0)
        )
        artifacts["mode"] = replace_mode

        return DeidResult(
            text=clean_text,
            entities=entities,
            replacement_map=replacement_map,
            events=events,
            timings_ms=timings_ms,
            artifacts=artifacts,
        )

    @staticmethod
    def _normalize_output_mode(output_mode: str) -> str:
        mode = (output_mode or "").strip().lower()
        if mode in {"replacement", "replace"}:
            return "replace"
        if mode in {"blackbox", "black", "redact", "mask"}:
            return "blackbox"
        return "replace"


__all__ = ["DeidPipeline", "DeidResult"]
