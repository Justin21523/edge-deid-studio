# src/deid_pipeline/pii/detectors/composite.py
from __future__ import annotations

from typing import List, Sequence, Tuple

from ...config import Config
from ..utils import logger
from ..utils.base import Entity, PIIDetector

class CompositeDetector(PIIDetector):
    def __init__(self, *detectors: PIIDetector):
        self.detectors = detectors
        self.config = Config()

    def detect(self, text: str) -> List[Entity]:
        all_ents = []

        # Execute detectors sequentially (can be parallelized in future).
        for detector in self.detectors:
            try:
                ents = detector.detect(text)
                all_ents.extend(ents)
                logger.debug(
                    "%s found %d entities", detector.__class__.__name__, len(ents)
                )
            except Exception as e:
                logger.error(
                    "%s detection failed: %s", detector.__class__.__name__, str(e)
                )

        # Resolve entity conflicts.
        resolved_ents = self._resolve_conflicts(all_ents)
        return resolved_ents

    def _resolve_conflicts(self, entities: List[Entity]) -> List[Entity]:
        """
        Resolve overlapping entities using ENTITY_PRIORITY.

        Rules:
        1) Sort by start offset.
        2) If overlap ratio > threshold, keep the entity with higher priority;
           tie-break by score, then by shorter source string.
        3) If overlap ratio <= threshold, keep both.
        """
        if not entities:
            return []

        overlap_threshold = 0.5

        def span_tuple(entity: Entity) -> Tuple[int, int]:
            start, end = entity["span"]
            return int(start), int(end)

        sorted_ents = sorted(entities, key=lambda e: span_tuple(e)[0])
        resolved: List[Entity] = []

        for entity in sorted_ents:
            if not resolved:
                resolved.append(entity)
                continue

            last = resolved[-1]
            last_start, last_end = span_tuple(last)
            cur_start, cur_end = span_tuple(entity)

            if cur_start >= last_end:
                resolved.append(entity)
                continue

            overlap = max(0, min(last_end, cur_end) - cur_start)
            last_len = max(1, last_end - last_start)
            cur_len = max(1, cur_end - cur_start)
            overlap_ratio = overlap / min(last_len, cur_len)

            if overlap_ratio <= overlap_threshold:
                resolved.append(entity)
                continue

            winner = self._pick_winner(last, entity)
            if winner is entity:
                resolved[-1] = entity

        return resolved


    def _pick_winner(self, a: Entity, b: Entity) -> Entity:
        a_priority = self._get_priority(a["type"])
        b_priority = self._get_priority(b["type"])

        if a_priority != b_priority:
            return a if a_priority > b_priority else b

        if a["score"] != b["score"]:
            return a if a["score"] > b["score"] else b

        a_source = a.get("source", "")
        b_source = b.get("source", "")
        if len(a_source) != len(b_source):
            return a if len(a_source) < len(b_source) else b

        return a

    def _get_priority(self, entity_type: str) -> int:
        """Return the priority for a given entity type."""

        return self.config.ENTITY_PRIORITY.get(entity_type, 50)
