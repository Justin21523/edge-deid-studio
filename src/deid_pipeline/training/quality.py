from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.contracts import Entity
from .datasets import SpanExample


def summarize_numbers(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0}

    sorted_values = sorted(float(v) for v in values)
    p50 = statistics.median(sorted_values)
    p95 = sorted_values[int(0.95 * (len(sorted_values) - 1))]
    return {
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "mean": float(sum(sorted_values) / len(sorted_values)),
        "p50": float(p50),
        "p95": float(p95),
    }


def _safe_span(span: Any) -> Optional[Tuple[int, int]]:
    if not span:
        return None
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            return int(span[0]), int(span[1])
        except Exception:
            return None
    return None


def validate_span_example(text: str, entities: Sequence[Entity]) -> List[Dict[str, Any]]:
    """Validate a span example and return a list of issue dictionaries."""

    issues: List[Dict[str, Any]] = []

    spans_with_indices: List[Tuple[int, int, int]] = []

    for idx, ent in enumerate(entities):
        ent_type = ent.get("type")
        if not ent_type:
            issues.append({"kind": "missing_type", "entity_index": idx})

        span = _safe_span(ent.get("span"))
        if span is None:
            issues.append({"kind": "invalid_span", "entity_index": idx, "span": ent.get("span")})
            continue

        start, end = span
        if start < 0 or end < 0 or start > len(text) or end > len(text):
            issues.append(
                {
                    "kind": "span_out_of_bounds",
                    "entity_index": idx,
                    "type": ent_type,
                    "span": [start, end],
                    "text_len": len(text),
                }
            )
            continue

        if start >= end:
            issues.append(
                {
                    "kind": "empty_span",
                    "entity_index": idx,
                    "type": ent_type,
                    "span": [start, end],
                }
            )
            continue

        spans_with_indices.append((start, end, idx))

        expected = text[start:end]
        provided = ent.get("text")
        if provided is not None and str(provided) != expected:
            issues.append(
                {
                    "kind": "text_mismatch",
                    "entity_index": idx,
                    "type": ent_type,
                    "span": [start, end],
                }
            )

    spans_with_indices.sort(key=lambda x: (x[0], x[1]))
    for (a_start, a_end, a_idx), (b_start, b_end, b_idx) in zip(
        spans_with_indices,
        spans_with_indices[1:],
    ):
        if b_start < a_end:
            issues.append(
                {
                    "kind": "overlap",
                    "entity_a_index": a_idx,
                    "entity_b_index": b_idx,
                    "span_a": [a_start, a_end],
                    "span_b": [b_start, b_end],
                }
            )

    return issues


def analyze_span_examples(
    examples: Sequence[SpanExample],
    *,
    max_issues: int = 200,
) -> Dict[str, Any]:
    """Analyze a span dataset and return a JSON-serializable quality report."""

    issue_counts: Dict[str, int] = {}
    issues: List[Dict[str, Any]] = []

    entity_type_counts: Dict[str, int] = {}
    text_lengths: List[float] = []
    entity_lengths: List[float] = []
    entities_per_example: List[float] = []

    invalid_examples = 0
    invalid_entities = 0
    total_entities = 0

    for example_index, ex in enumerate(examples):
        text = str(ex.text or "")
        ents = list(ex.entities or ())
        text_lengths.append(float(len(text)))
        entities_per_example.append(float(len(ents)))

        total_entities += len(ents)

        ex_issues = validate_span_example(text, ents)
        if ex_issues:
            invalid_examples += 1
            for issue in ex_issues:
                kind = str(issue.get("kind", "unknown"))
                issue_counts[kind] = int(issue_counts.get(kind, 0)) + 1
                if len(issues) < int(max_issues):
                    issues.append({"example_index": example_index, **issue})

        for idx, ent in enumerate(ents):
            ent_type = ent.get("type")
            if ent_type:
                entity_type_counts[str(ent_type)] = int(entity_type_counts.get(str(ent_type), 0)) + 1

            span = _safe_span(ent.get("span"))
            if span is None:
                invalid_entities += 1
                continue
            start, end = span
            if start < 0 or end < 0 or start > len(text) or end > len(text) or start >= end:
                invalid_entities += 1
                continue
            entity_lengths.append(float(end - start))

    report: Dict[str, Any] = {
        "summary": {
            "examples": int(len(examples)),
            "entities": int(total_entities),
            "invalid_examples": int(invalid_examples),
            "invalid_entities": int(invalid_entities),
            "issue_counts": dict(sorted(issue_counts.items(), key=lambda x: (-x[1], x[0]))),
            "entity_type_counts": dict(sorted(entity_type_counts.items(), key=lambda x: (-x[1], x[0]))),
            "text_length": summarize_numbers(text_lengths),
            "entities_per_example": summarize_numbers(entities_per_example),
            "entity_span_length": summarize_numbers(entity_lengths),
        },
        "issues": issues,
    }

    return report

