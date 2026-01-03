from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

from .base import Entity
from .fake_provider import FakeProvider
from ...replace.cache import DEFAULT_REPLACEMENT_CACHE, ReplacementCache

class Replacer:
    def __init__(self, provider=None, cache: Optional[ReplacementCache[str, str]] = None):
        self.provider = provider or FakeProvider()
        self.cache = cache or DEFAULT_REPLACEMENT_CACHE

    def replace(
        self,
        text: str,
        entities: List[Entity],
        mode: str = "replace",
        *,
        context_hash: Optional[str] = None,
    ) -> Tuple[str, List[dict]]:
        """Replace or mask PII entities in the given text.

        Args:
            text: Original text.
            entities: Detected entities (span offsets refer to the original text).
            mode: "replace" or "blackbox" (aliases: "black", "redact", "mask").
            context_hash: A stable hash used for deterministic replacements within a document.

        Returns:
            (new_text, events)
        """

        normalized = (mode or "").strip().lower()
        if normalized in {"blackbox", "black", "redact", "mask"}:
            return self._blackbox_mode(text, entities)
        return self._replace_mode(text, entities, context_hash=context_hash)

    def _replace_mode(
        self, text: str, entities: List[Entity], *, context_hash: Optional[str]
    ) -> Tuple[str, List[dict]]:
        """Replace entities with generated fake values."""

        doc_context_hash = context_hash or hashlib.sha256(
            text.encode("utf-8", errors="replace")
        ).hexdigest()

        sorted_ents = sorted(entities, key=lambda x: x["span"][0], reverse=True)
        new_text = text
        events = []

        for ent in sorted_ents:
            start, end = ent["span"]
            original = text[start:end]

            entity_type = str(ent["type"])
            cache_key = f"{entity_type}:{original}:{doc_context_hash}"

            def _factory() -> str:
                if hasattr(self.provider, "generate_deterministic"):
                    return self.provider.generate_deterministic(  # type: ignore[attr-defined]
                        entity_type, original, context_hash=doc_context_hash
                    )
                return self.provider.generate(entity_type, original)

            replacement = self.cache.get_or_set(cache_key, _factory)

            new_text = new_text[:start] + replacement + new_text[end:]

            events.append({
                "original": original,
                "replacement": replacement,
                "fake": replacement,  # backward-compatible alias
                "type": entity_type,
                "span": (start, start + len(replacement)),
                "source": ent.get("source", "unknown")
            })

        return new_text, events

    def _blackbox_mode(self, text: str, entities: List[Entity]) -> Tuple[str, List[dict]]:
        """Mask entities using fixed-width block characters."""

        sorted_ents = sorted(entities, key=lambda x: x["span"][0], reverse=True)
        new_text = text
        events = []

        for ent in sorted_ents:
            start, end = ent["span"]

            blackbox = "█" * (end - start)
            new_text = new_text[:start] + blackbox + new_text[end:]

            events.append({
                "type": ent["type"],
                "span": (start, start + len(blackbox)),
                "source": ent.get("source", "unknown")
            })

        return new_text, events

    @staticmethod
    def dumps(events: List[dict]) -> str:
        """JSON helper used by CLI examples."""

        return json.dumps(events, ensure_ascii=False, indent=2)
