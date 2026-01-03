from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.contracts import Entity


PLACEHOLDER_REGEX = re.compile(r"[<\[{]([A-Za-z_]+?)(?:_?\d+)?[>\]}]")


PLACEHOLDER_TYPE_MAP: Dict[str, str] = {
    "PER": "NAME",
    "PERSON": "NAME",
    "NAME": "NAME",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "MOBILE": "PHONE",
    "TEL": "PHONE",
    "ADDRESS": "ADDRESS",
    "LOCATION": "ADDRESS",
    "LOC": "ADDRESS",
    "GPE": "ADDRESS",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "ID": "ID",
    "SSN": "ID",
    "PASSPORT": "PASSPORT",
    "MEDICAL_ID": "MEDICAL_ID",
    "UNIFIED_BUSINESS_NO": "UNIFIED_BUSINESS_NO",
}


@dataclass(frozen=True)
class TokenSpan:
    token: str
    start: int
    end: int


def tokenize_whitespace(text: str) -> List[TokenSpan]:
    """Tokenize by whitespace and return tokens with character offsets."""

    spans: List[TokenSpan] = []
    for match in re.finditer(r"\S+", text):
        spans.append(TokenSpan(token=match.group(0), start=match.start(), end=match.end()))
    return spans


def parse_placeholder_type(token: str) -> Optional[str]:
    """Parse a placeholder token and return a canonical entity type when possible."""

    match = PLACEHOLDER_REGEX.search(token)
    if not match:
        return None
    raw = match.group(1).strip().upper()
    if not raw:
        return None
    return PLACEHOLDER_TYPE_MAP.get(raw, raw)


def extract_entities_from_masked_pair(
    original_text: str,
    masked_text: str,
    *,
    language: str = "en",
    source: str = "masked_pair",
) -> List[Entity]:
    """Extract gold spans from an (original, masked) text pair.

    Assumptions:
    - `masked_text` contains placeholders like `<NAME>` / `[EMAIL]` / `{PHONE}`.
    - The placeholder indicates the entity type for the replaced text span.
    """

    original_tokens = tokenize_whitespace(original_text)
    masked_tokens = tokenize_whitespace(masked_text)

    a = [t.token for t in original_tokens]
    b = [t.token for t in masked_tokens]

    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    entities: List[Entity] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        if i1 >= i2:
            continue

        placeholder_types: List[str] = []
        for tok in masked_tokens[j1:j2]:
            parsed = parse_placeholder_type(tok.token)
            if parsed:
                placeholder_types.append(parsed)

        if not placeholder_types:
            continue

        # If we have a 1:1 mapping (token count equals placeholder count), emit per-token entities.
        if (i2 - i1) == len(placeholder_types):
            for offset, entity_type in enumerate(placeholder_types):
                orig_tok = original_tokens[i1 + offset]
                span = (orig_tok.start, orig_tok.end)
                entities.append(
                    {
                        "type": str(entity_type),
                        "span": span,
                        "text": original_text[span[0] : span[1]],
                        "confidence": 1.0,
                        "score": 1.0,
                        "source": source,
                        "language": language,
                    }
                )
            continue

        # Otherwise, use a single entity covering the entire replaced span.
        unique_types = sorted(set(placeholder_types))
        entity_type = unique_types[0] if len(unique_types) == 1 else "PII"
        start = original_tokens[i1].start
        end = original_tokens[i2 - 1].end
        span = (start, end)
        entities.append(
            {
                "type": str(entity_type),
                "span": span,
                "text": original_text[start:end],
                "confidence": 1.0,
                "score": 1.0,
                "source": source,
                "language": language,
            }
        )

    return _dedupe_entities(entities)


def _dedupe_entities(entities: Sequence[Entity]) -> List[Entity]:
    seen = set()
    deduped: List[Entity] = []
    for ent in entities:
        span = ent.get("span")
        ent_type = ent.get("type")
        if not span or not ent_type:
            continue
        key = (int(span[0]), int(span[1]), str(ent_type))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ent)
    return deduped
