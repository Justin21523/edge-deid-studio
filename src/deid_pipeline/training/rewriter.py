from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Protocol

from .canonical import canonicalize_entity_type
from .placeholders import PLACEHOLDER_RE


class DeterministicRewriterProvider(Protocol):
    def generate_deterministic(self, entity_type: str, original: str, *, context_hash: str) -> str: ...


PUNCT_SWAP: Dict[str, str] = {
    "，": ",",
    ",": "，",
    "。": ".",
    ".": "。",
    "：": ":",
    ":": "：",
    "；": ";",
    ";": "；",
}


def hash16(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def fill_placeholders_with_fake_values(
    text: str,
    provider: DeterministicRewriterProvider,
    *,
    context_hash: str,
    unknown_entity_type: str = "PII",
) -> str:
    """Fill `<TYPE>` placeholders with deterministic fake values (offline/local-only).

    Determinism contract: outputs must be stable for the same triple:
    (entity_type, original_value, context_hash).
    """

    raw = str(text or "")
    matches = list(PLACEHOLDER_RE.finditer(raw))
    if not matches:
        return raw

    parts: List[str] = []
    cursor = 0
    for idx, match in enumerate(matches):
        parts.append(raw[cursor : match.start()])

        raw_type = str(match.group(2) or "")
        ent_type = canonicalize_entity_type(raw_type, unknown_label=unknown_entity_type) or unknown_entity_type
        original = f"{raw_type}:{idx}"
        value = provider.generate_deterministic(str(ent_type), original, context_hash=str(context_hash))

        parts.append(str(value))
        cursor = match.end()

    parts.append(raw[cursor:])
    return "".join(parts)


def make_noisy(
    text: str,
    rng: random.Random,
    *,
    punct_prob: float = 0.35,
    space_prob: float = 0.25,
    dup_prob: float = 0.08,
) -> str:
    """Create a noisy version of the text without altering alnum/CJK tokens."""

    out: List[str] = []
    for ch in str(text or ""):
        if ch in PUNCT_SWAP and rng.random() < float(punct_prob):
            ch = PUNCT_SWAP[ch]
        out.append(ch)
        if ch in PUNCT_SWAP and rng.random() < float(space_prob):
            out.append(" ")
        if ch in PUNCT_SWAP and rng.random() < float(dup_prob):
            out.append(ch)

    noisy = "".join(out)
    noisy = noisy.replace("  ", " ") if rng.random() < 0.5 else noisy.replace(" ", "  ")
    return noisy.strip()


def build_prompt(noisy: str, clean: str, *, language: str) -> str:
    if language == "zh":
        return (
            "Rewrite the following de-identified Chinese text to be fluent and natural. "
            "Preserve all replacement values exactly.\n"
            f"INPUT: {noisy}\n"
            f"OUTPUT: {clean}\n"
        )
    return (
        "Rewrite the following de-identified text to be fluent and natural. "
        "Preserve all replacement values exactly.\n"
        f"INPUT: {noisy}\n"
        f"OUTPUT: {clean}\n"
    )

