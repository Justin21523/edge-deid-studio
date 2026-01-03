from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Sequence


MarkupFormat = Literal["xml", "bracket"]


XML_PII_RE = re.compile(
    r"<PII\s+type=['\"]([A-Za-z0-9_]+)['\"]\s*>(.*?)</PII>",
    flags=re.DOTALL,
)
BRACKET_PII_RE = re.compile(
    r"\[([A-Za-z0-9_]+)\](.*?)\[/\1\]",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class PiiBlock:
    """A PII-marked block extracted from the input text."""

    pii_type: str
    value: str
    value_span: tuple[int, int]
    block_span: tuple[int, int]


def detect_markup_format(text: str) -> MarkupFormat | None:
    """Detect which markup format is present in the text."""

    raw = str(text or "")
    if XML_PII_RE.search(raw):
        return "xml"
    if BRACKET_PII_RE.search(raw):
        return "bracket"
    return None


def iter_pii_blocks(text: str, *, fmt: MarkupFormat) -> Iterator[PiiBlock]:
    raw = str(text or "")
    regex = XML_PII_RE if fmt == "xml" else BRACKET_PII_RE
    for match in regex.finditer(raw):
        pii_type = str(match.group(1) or "").strip()
        value = str(match.group(2) or "")
        yield PiiBlock(
            pii_type=pii_type,
            value=value,
            value_span=(int(match.start(2)), int(match.end(2))),
            block_span=(int(match.start(0)), int(match.end(0))),
        )


def extract_pii_blocks(text: str, *, fmt: MarkupFormat | None = None) -> list[PiiBlock]:
    """Extract PII-marked blocks from the text.

    When fmt is None, the function auto-detects the markup.
    """

    resolved = fmt or detect_markup_format(text)
    if resolved is None:
        return []
    return list(iter_pii_blocks(text, fmt=resolved))


def mask_spans(text: str, spans: Sequence[tuple[int, int]], *, placeholder: str = "") -> str:
    """Mask a set of spans by replacing them with a placeholder (default: delete)."""

    raw = str(text or "")
    if not spans:
        return raw

    safe: list[tuple[int, int]] = []
    for start, end in spans:
        s, e = int(start), int(end)
        if s < 0 or e <= s or e > len(raw):
            continue
        safe.append((s, e))
    if not safe:
        return raw

    safe.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = raw
    for start, end in safe:
        out = out[:start] + str(placeholder) + out[end:]
    return out


def mask_pii_blocks(text: str, blocks: Iterable[PiiBlock], *, placeholder: str = "") -> str:
    """Mask full PII blocks (including tag wrappers) in the text."""

    spans = [b.block_span for b in blocks]
    return mask_spans(str(text or ""), spans, placeholder=placeholder)

