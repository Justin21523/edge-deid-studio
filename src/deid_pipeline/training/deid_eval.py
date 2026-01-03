from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .canonical import canonicalize_entity_type
from .pii_markup import PiiBlock, extract_pii_blocks, mask_pii_blocks


@dataclass(frozen=True)
class DeidExample:
    example_id: str
    input_text: str
    target_text: str


@dataclass(frozen=True)
class DeidPrediction:
    example_id: str
    prediction_text: str


def iter_deid_examples_jsonl(path: str | Path) -> Iterator[DeidExample]:
    """Iterate `{id,input,output}` JSONL records without loading the full file."""

    data_path = Path(path).expanduser().resolve()
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = (line or "").strip()
            if not raw:
                continue
            row = json.loads(raw)
            ex_id = str(row.get("id", "") or row.get("_id", "") or row.get("example_id", "") or "")
            if not ex_id:
                ex_id = str(row.get("uuid", "") or row.get("pk", "") or "")
            if not ex_id:
                # Fall back to a stable hash-like surrogate.
                ex_id = str(row.get("index", "") or "")
            yield DeidExample(
                example_id=ex_id,
                input_text=str(row.get("input", "") or ""),
                target_text=str(row.get("output", "") or ""),
            )


def iter_predictions_jsonl(path: str | Path) -> Iterator[DeidPrediction]:
    """Iterate `{id,prediction}` JSONL records."""

    pred_path = Path(path).expanduser().resolve()
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = (line or "").strip()
            if not raw:
                continue
            row = json.loads(raw)
            ex_id = str(row.get("id", "") or row.get("example_id", "") or row.get("_id", "") or "")
            yield DeidPrediction(example_id=ex_id, prediction_text=str(row.get("prediction", "") or row.get("output", "") or ""))


def load_banned_phrases(path: str | Path) -> List[str]:
    """Load banned output phrases from YAML or JSON."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Banned phrases config not found: {config_path}")

    raw = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    data: Any
    if suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
    elif suffix == ".json":
        data = json.loads(raw)
    else:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(raw)
        except Exception:
            data = json.loads(raw)

    if data is None:
        return []
    if isinstance(data, dict):
        items = data.get("banned_phrases") or data.get("phrases") or data.get("banned") or []
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError(f"Invalid banned phrases config (expected list): {config_path}")
    return [str(x) for x in items if str(x).strip()]


TYPE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(?:\+?\d{1,3}[-\s]?)?(?:0?9\d{8}|\d{2,4}[-\s]?\d{3,4}[-\s]?\d{3,4})"),
    "ID": re.compile(r"[A-Z][12]\d{8}"),  # Taiwan ID proxy (TW_ID canonicalizes to ID)
    "UNIFIED_BUSINESS_NO": re.compile(r"\b\d{8}\b"),
    "BANK_ACCOUNT": re.compile(r"\b\d{10,20}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "LICENSE_PLATE": re.compile(r"\b[A-Z]{2,3}-?\d{3,4}\b"),
    "PASSPORT": re.compile(r"\b[A-Z]{1,2}\d{7,8}\b"),
    "MEDICAL_ID": re.compile(r"\bM\d{7}\b"),
    "URL": re.compile(r"https?://[^\s]+"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "DATE": re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
}


def _count_type_matches(text: str, *, pii_type: str) -> int:
    pattern = TYPE_PATTERNS.get(str(pii_type))
    if pattern is None:
        return 0
    return len(pattern.findall(str(text or "")))


def compute_pii_removal_recall(pii_values: Sequence[str], prediction_text: str) -> Tuple[float, int, int]:
    """Return (recall, removed_count, total_count)."""

    pred = str(prediction_text or "")
    total = int(len(pii_values))
    if total == 0:
        return 1.0, 0, 0
    removed = 0
    for value in pii_values:
        v = str(value or "")
        if not v:
            continue
        if v not in pred:
            removed += 1
    recall = float(removed) / float(total) if total > 0 else 1.0
    return recall, int(removed), int(total)


def sequence_similarity(a: str, b: str) -> float:
    """Return a fast similarity ratio in [0,1] (SequenceMatcher proxy)."""

    return float(SequenceMatcher(None, str(a or ""), str(b or "")).ratio())


def compute_type_consistency(
    pii_blocks: Sequence[PiiBlock],
    prediction_text: str,
    *,
    unknown_label: str = "PII",
) -> Tuple[float, Dict[str, Dict[str, int]]]:
    """Compute a type-consistency proxy based on regex match counts.

    Returns:
    - macro consistency in [0,1]
    - per-type details: {type: {expected, found}}
    """

    expected: Dict[str, int] = {}
    for block in pii_blocks:
        canon = canonicalize_entity_type(block.pii_type, unknown_label=unknown_label) or unknown_label
        expected[str(canon)] = int(expected.get(str(canon), 0) + 1)

    details: Dict[str, Dict[str, int]] = {}
    scores: List[float] = []
    for pii_type, exp_count in expected.items():
        if pii_type not in TYPE_PATTERNS:
            continue
        found = _count_type_matches(prediction_text, pii_type=pii_type)
        details[pii_type] = {"expected": int(exp_count), "found": int(found)}
        if int(exp_count) <= 0:
            continue
        scores.append(min(1.0, float(found) / float(exp_count)))

    if not scores:
        return 1.0, details
    return float(sum(scores) / float(len(scores))), details


def check_format_compliance(prediction_text: str, banned_phrases: Sequence[str]) -> Tuple[bool, List[str]]:
    """Return (is_compliant, triggered_phrases)."""

    text = str(prediction_text or "").strip()
    if not text:
        return False, ["<EMPTY>"]

    lowered = text.lower()
    triggered: List[str] = []

    # Always disallow common prompt separators.
    for needle in ["input:", "output:", "analysis:", "explanation:", "step", "reason:"]:
        if needle in lowered:
            triggered.append(needle)

    # Disallow leaving markup in the output.
    if "<pii" in lowered or "[/" in lowered:
        triggered.append("<PII>/<[/...]>")

    for phrase in banned_phrases:
        p = str(phrase or "").strip()
        if not p:
            continue
        if p.lower() in lowered:
            triggered.append(p)

    return len(triggered) == 0, triggered


def detect_pii_blocks(
    input_text: str,
    *,
    prefer_markup: bool = True,
    regex_spans: Optional[Sequence[Tuple[int, int, str]]] = None,
) -> List[PiiBlock]:
    """Detect PII blocks from markup or weak-label spans."""

    if prefer_markup:
        blocks = extract_pii_blocks(input_text, fmt=None)
        if blocks:
            return blocks

    if regex_spans:
        blocks: List[PiiBlock] = []
        raw = str(input_text or "")
        for idx, (start, end, pii_type) in enumerate(list(regex_spans)):
            s, e = int(start), int(end)
            if s < 0 or e <= s or e > len(raw):
                continue
            blocks.append(
                PiiBlock(
                    pii_type=str(pii_type),
                    value=raw[s:e],
                    value_span=(s, e),
                    block_span=(s, e),
                )
            )
        return blocks

    return []


@dataclass(frozen=True)
class ExampleReport:
    example_id: str
    pii_removal_recall: float
    pii_total: int
    pii_leak_count: int
    non_pii_similarity: float
    over_rewrite_rate: float
    type_consistency: float
    type_details: Dict[str, Dict[str, int]]
    repetition_3gram_rate: float
    format_compliant: bool
    format_triggers: List[str]


def repetition_3gram_rate(text: str) -> float:
    raw = str(text or "")
    if len(raw) < 3:
        return 0.0
    grams = [raw[i : i + 3] for i in range(0, len(raw) - 2)]
    if not grams:
        return 0.0
    counts: Dict[str, int] = {}
    for g in grams:
        counts[g] = int(counts.get(g, 0) + 1)
    repeated = sum(v for v in counts.values() if int(v) > 1)
    return float(repeated) / float(len(grams))


def evaluate_prediction(
    example: DeidExample,
    prediction_text: str,
    *,
    pii_blocks: Sequence[PiiBlock],
    banned_phrases: Sequence[str],
) -> ExampleReport:
    pii_values = [b.value for b in pii_blocks if str(b.value or "").strip()]
    pii_recall, removed, total = compute_pii_removal_recall(pii_values, prediction_text)
    leak = int(total - removed)

    masked_input = mask_pii_blocks(example.input_text, pii_blocks, placeholder="")
    masked_input = " ".join(masked_input.split())
    pred_clean = " ".join(str(prediction_text or "").split())
    sim = sequence_similarity(masked_input, pred_clean) if masked_input else 1.0
    over = float(1.0 - sim)

    type_consistency, type_details = compute_type_consistency(pii_blocks, prediction_text)
    rep_rate = repetition_3gram_rate(prediction_text)
    compliant, triggers = check_format_compliance(prediction_text, banned_phrases)

    return ExampleReport(
        example_id=str(example.example_id),
        pii_removal_recall=float(pii_recall),
        pii_total=int(total),
        pii_leak_count=int(leak),
        non_pii_similarity=float(sim),
        over_rewrite_rate=float(over),
        type_consistency=float(type_consistency),
        type_details=dict(type_details),
        repetition_3gram_rate=float(rep_rate),
        format_compliant=bool(compliant),
        format_triggers=list(triggers),
    )


def aggregate_reports(reports: Iterable[ExampleReport]) -> Dict[str, Any]:
    rows = list(reports)
    if not rows:
        return {
            "count": 0,
            "pii_removal_recall": 1.0,
            "pii_leak_rate": 0.0,
            "over_rewrite_rate": 0.0,
            "type_consistency": 1.0,
            "repetition_3gram_rate": 0.0,
            "format_compliance_rate": 1.0,
        }

    total_pii = sum(int(r.pii_total) for r in rows)
    total_leaks = sum(int(r.pii_leak_count) for r in rows)
    leak_rate = float(total_leaks) / float(total_pii) if total_pii > 0 else 0.0

    def avg(values: Sequence[float]) -> float:
        return float(sum(values) / float(len(values))) if values else 0.0

    return {
        "count": int(len(rows)),
        "pii_removal_recall": avg([float(r.pii_removal_recall) for r in rows]),
        "pii_total": int(total_pii),
        "pii_leak_count": int(total_leaks),
        "pii_leak_rate": float(leak_rate),
        "over_rewrite_rate": avg([float(r.over_rewrite_rate) for r in rows]),
        "type_consistency": avg([float(r.type_consistency) for r in rows]),
        "repetition_3gram_rate": avg([float(r.repetition_3gram_rate) for r in rows]),
        "format_compliance_rate": avg([1.0 if bool(r.format_compliant) else 0.0 for r in rows]),
    }
