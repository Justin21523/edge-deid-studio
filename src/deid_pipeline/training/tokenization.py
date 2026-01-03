from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from ..core.contracts import Entity


def build_bio_label_list(entity_types: Iterable[str]) -> List[str]:
    """Build a BIO label list suitable for token classification training."""

    types = sorted({str(t) for t in entity_types if t})
    labels = ["O"]
    for t in types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    return labels


def align_entities_to_tokens(
    *,
    entities: Sequence[Entity],
    offset_mapping: Sequence[Tuple[int, int]],
    label_to_id: Dict[str, int],
) -> List[int]:
    """Align entity spans to token offsets and return label ids."""

    labels: List[int] = []
    for tok_start, tok_end in offset_mapping:
        if tok_start == tok_end == 0:
            labels.append(-100)
            continue

        label = "O"
        for ent in entities:
            span = ent.get("span")
            ent_type = ent.get("type")
            if not span or not ent_type:
                continue
            ent_start, ent_end = span
            if tok_start >= ent_start and tok_end <= ent_end:
                prefix = "B-" if tok_start == ent_start else "I-"
                label = f"{prefix}{ent_type}"
                break

        labels.append(int(label_to_id.get(label, label_to_id["O"])))

    return labels

