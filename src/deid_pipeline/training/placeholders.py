from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from ..core.contracts import Entity
from .canonical import canonicalize_entity_type


PLACEHOLDER_RE = re.compile(r"([<\[{])([A-Za-z_]+?)(?:_?\d+)?([>\]}])")


def contains_cjk(text: str) -> bool:
    """Return True when the text contains CJK characters (rough heuristic)."""

    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            return True
    return False


def canonicalize_placeholder_text(text: str, *, unknown_label: str = "PII") -> str:
    """Canonicalize placeholder tokens inside text (e.g. `<LASTNAME_1>` → `<NAME>`)."""

    def repl(match: re.Match[str]) -> str:
        left, raw_type, right = match.group(1), match.group(2), match.group(3)
        mapped = canonicalize_entity_type(raw_type, unknown_label=unknown_label)
        return f"{left}{mapped or unknown_label}{right}"

    return PLACEHOLDER_RE.sub(repl, text or "")


def replace_spans_with_placeholders(
    text: str,
    entities: Sequence[Entity],
    *,
    unknown_label: str = "PII",
) -> str:
    """Replace entity spans with canonical placeholders (e.g. `<EMAIL>`).

    This is used to create a GPT-2 fine-tuning corpus that is safe to train on.
    """

    if not entities:
        return text

    spans: List[Tuple[int, int, str]] = []
    for ent in entities:
        span = ent.get("span")
        if not span:
            continue
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start or end > len(text):
            continue
        ent_type = canonicalize_entity_type(ent.get("type"), unknown_label=unknown_label)
        if not ent_type:
            continue
        spans.append((start, end, str(ent_type)))

    if not spans:
        return text

    # Replace from the end to keep offsets stable.
    spans.sort(key=lambda x: (x[0], x[1]), reverse=True)

    out = text
    for start, end, ent_type in spans:
        out = out[:start] + f"<{ent_type}>" + out[end:]
    return out

