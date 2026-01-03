from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple

from ..core.contracts import Entity
from .masked_pairs import extract_entities_from_masked_pair


CANONICAL_ENTITY_MAP: Dict[str, str] = {
    "PER": "NAME",
    "PERSON": "NAME",
    "LOC": "ADDRESS",
    "GPE": "ADDRESS",
    "ORG": "ORGANIZATION",
}

WIKIANN_ID2LABEL: Tuple[str, ...] = (
    "B-LOC",
    "B-ORG",
    "B-PER",
    "I-LOC",
    "I-ORG",
    "I-PER",
    "O",
)


def _require_network(allow_network: bool, *, dataset_name: str) -> None:
    if allow_network:
        return
    if Path(dataset_name).expanduser().exists():
        return
    raise RuntimeError(
        "Network access is disabled by default for training/data tooling. "
        "Pass allow_network=True explicitly when you intend to download datasets."
    )


@dataclass(frozen=True)
class TokenNERExample:
    """Token-level NER example (BIO tags)."""

    tokens: Tuple[str, ...]
    tags: Tuple[str, ...]


class DatasetAdapter(Protocol):
    """Adapter contract that converts a dataset into token-level NER examples."""

    def load(self, *, split: str, allow_network: bool = False) -> Sequence[TokenNERExample]: ...


@dataclass(frozen=True)
class SpanExample:
    """Span-based NER example (gold character offsets)."""

    text: str
    entities: Tuple[Entity, ...]


class SpanDatasetAdapter(Protocol):
    """Adapter contract that converts a dataset into span-based NER examples."""

    def load_span_examples(self, *, split: str, allow_network: bool = False) -> Sequence[SpanExample]: ...


def normalize_bio_tag(tag: str) -> str:
    """Normalize BIO tags to the canonical entity set."""

    raw = (tag or "O").strip()
    if raw == "O":
        return "O"

    prefix = "B-" if raw.startswith("B-") else "I-" if raw.startswith("I-") else ""
    base = raw[len(prefix) :] if prefix else raw
    mapped = CANONICAL_ENTITY_MAP.get(base, base)
    return f"{prefix}{mapped}" if prefix else mapped


class HuggingFaceTokenNERAdapter:
    """Adapter for HF datasets that expose `tokens` and `ner_tags` fields."""

    def __init__(
        self,
        dataset_name: str,
        *,
        config_name: Optional[str] = None,
        tokens_field: str = "tokens",
        tags_field: str = "ner_tags",
        tag_id_to_label: Optional[Sequence[str]] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.tokens_field = tokens_field
        self.tags_field = tags_field
        self.tag_id_to_label = tuple(tag_id_to_label) if tag_id_to_label is not None else None

    def load(
        self,
        *,
        split: str,
        allow_network: bool = False,
        trust_remote_code: bool = False,
    ) -> Sequence[TokenNERExample]:
        return list(
            self.iter_load(
                split=split,
                allow_network=allow_network,
                trust_remote_code=trust_remote_code,
            )
        )

    def iter_load(
        self,
        *,
        split: str,
        allow_network: bool = False,
        trust_remote_code: bool = False,
    ) -> Iterator[TokenNERExample]:
        """Stream token-level NER examples without materializing the full dataset."""

        _require_network(allow_network, dataset_name=self.dataset_name)

        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            self.dataset_name,
            self.config_name,
            split=split,
            trust_remote_code=bool(trust_remote_code),
        )
        columns = set(ds.column_names)
        tokens_field = self.tokens_field if self.tokens_field in columns else "tokens" if "tokens" in columns else None
        tags_field = (
            self.tags_field
            if self.tags_field in columns
            else "ner_tags"
            if "ner_tags" in columns
            else "tags"
            if "tags" in columns
            else None
        )
        if tokens_field is None or tags_field is None:
            raise KeyError(
                "Unable to infer token/tag fields. "
                f"columns={sorted(columns)} tokens_field={self.tokens_field} tags_field={self.tags_field}"
            )

        tag_names: Optional[Sequence[str]] = None
        feature = ds.features[tags_field]
        tag_feature = getattr(feature, "feature", feature)
        names = getattr(tag_feature, "names", None)
        if names:
            tag_names = list(names)
        elif self.tag_id_to_label:
            tag_names = list(self.tag_id_to_label)
        elif self.dataset_name == "tner/wikiann":
            tag_names = list(WIKIANN_ID2LABEL)

        for row in ds:
            tokens = tuple(str(t) for t in row[tokens_field])
            raw = row[tags_field]
            if not raw:
                continue

            if isinstance(raw[0], str):
                tags_raw = [str(t) for t in raw]
            else:
                if tag_names is None:
                    raise RuntimeError(
                        "Tag ids do not have an associated label mapping. "
                        "Provide `tag_id_to_label` for this dataset."
                    )
                tags_raw = [tag_names[int(i)] for i in raw]

            tags = tuple(normalize_bio_tag(t) for t in tags_raw)
            yield TokenNERExample(tokens=tokens, tags=tags)


def iter_token_examples_to_span_examples(
    examples: Iterable[TokenNERExample],
    *,
    separator: str = " ",
    language: str = "en",
    source: str = "token_ner",
) -> Iterator[SpanExample]:
    """Convert BIO token-level examples into span examples by joining tokens with a separator."""

    for example in examples:
        parts: List[str] = []
        token_spans: List[Tuple[int, int]] = []
        cursor = 0
        for idx, token in enumerate(example.tokens):
            if idx:
                parts.append(separator)
                cursor += len(separator)
            start = cursor
            parts.append(token)
            cursor += len(token)
            end = cursor
            token_spans.append((start, end))

        text = "".join(parts)

        entities: List[Entity] = []
        current_type: Optional[str] = None
        current_start: Optional[int] = None
        current_end: Optional[int] = None

        for tag, (tok_start, tok_end) in zip(example.tags, token_spans):
            if tag == "O":
                if current_type is not None and current_start is not None and current_end is not None:
                    entities.append(
                        {
                            "type": current_type,
                            "span": (current_start, current_end),
                            "text": text[current_start:current_end],
                            "confidence": 1.0,
                            "score": 1.0,
                            "source": source,
                            "language": language,
                        }
                    )
                current_type = None
                current_start = None
                current_end = None
                continue

            prefix = "B" if tag.startswith("B-") else "I" if tag.startswith("I-") else ""
            ent_type = tag.split("-", 1)[-1] if "-" in tag else tag

            if prefix == "B" or current_type != ent_type:
                if current_type is not None and current_start is not None and current_end is not None:
                    entities.append(
                        {
                            "type": current_type,
                            "span": (current_start, current_end),
                            "text": text[current_start:current_end],
                            "confidence": 1.0,
                            "score": 1.0,
                            "source": source,
                            "language": language,
                        }
                    )
                current_type = ent_type
                current_start = tok_start
                current_end = tok_end
            else:
                current_end = tok_end

        if current_type is not None and current_start is not None and current_end is not None:
            entities.append(
                {
                    "type": current_type,
                    "span": (current_start, current_end),
                    "text": text[current_start:current_end],
                    "confidence": 1.0,
                    "score": 1.0,
                    "source": source,
                    "language": language,
                }
            )

        yield SpanExample(text=text, entities=tuple(entities))


def token_examples_to_span_examples(
    examples: Sequence[TokenNERExample],
    *,
    separator: str = " ",
    language: str = "en",
    source: str = "token_ner",
) -> List[SpanExample]:
    """Convert BIO token-level examples into span examples by joining tokens with a separator."""

    return list(
        iter_token_examples_to_span_examples(
            examples,
            separator=separator,
            language=language,
            source=source,
        )
    )


def adapter_wikiann(*, language: str) -> HuggingFaceTokenNERAdapter:
    """tner/wikiann adapter (supports multiple languages via config_name)."""

    return HuggingFaceTokenNERAdapter(
        "tner/wikiann",
        config_name=str(language),
        tags_field="tags",
        tag_id_to_label=WIKIANN_ID2LABEL,
    )


def adapter_msra_ner() -> HuggingFaceTokenNERAdapter:
    """levow/msra_ner adapter."""

    return HuggingFaceTokenNERAdapter("levow/msra_ner")


def adapter_weibo_ner() -> HuggingFaceTokenNERAdapter:
    """hltcoe/weibo_ner adapter."""

    return HuggingFaceTokenNERAdapter("hltcoe/weibo_ner")


class HuggingFaceMaskedTextAdapter:
    """Adapter for datasets that provide an (original_text, masked_text) pair.

    The adapter extracts gold spans by diffing placeholders in the masked text (e.g. `<NAME>`).
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        config_name: Optional[str] = None,
        original_field_candidates: Sequence[str] = ("text", "original_text", "source", "input"),
        masked_field_candidates: Sequence[str] = ("masked_text", "anonymized_text", "target", "output"),
        language: str = "en",
    ) -> None:
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.original_field_candidates = tuple(original_field_candidates)
        self.masked_field_candidates = tuple(masked_field_candidates)
        self.language = language

    def load_span_examples(
        self,
        *,
        split: str,
        allow_network: bool = False,
        trust_remote_code: bool = False,
    ) -> Sequence[SpanExample]:
        return list(
            self.iter_span_examples(
                split=split,
                allow_network=allow_network,
                trust_remote_code=trust_remote_code,
            )
        )

    def iter_span_examples(
        self,
        *,
        split: str,
        allow_network: bool = False,
        trust_remote_code: bool = False,
    ) -> Iterator[SpanExample]:
        """Stream span examples without materializing the full dataset."""

        _require_network(allow_network, dataset_name=self.dataset_name)

        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            self.dataset_name,
            self.config_name,
            split=split,
            trust_remote_code=bool(trust_remote_code),
        )
        columns = set(ds.column_names)

        original_field = next((c for c in self.original_field_candidates if c in columns), None)
        masked_field = next((c for c in self.masked_field_candidates if c in columns), None)
        if original_field is None or masked_field is None:
            raise KeyError(
                "Unable to infer original/masked fields. "
                f"columns={sorted(columns)} "
                f"original_candidates={list(self.original_field_candidates)} "
                f"masked_candidates={list(self.masked_field_candidates)}"
            )

        for row in ds:
            original = str(row.get(original_field, "") or "")
            masked = str(row.get(masked_field, "") or "")
            if not original or not masked:
                continue
            entities = extract_entities_from_masked_pair(
                original,
                masked,
                language=self.language,
                source=self.dataset_name,
            )
            if not entities:
                continue
            yield SpanExample(text=original, entities=tuple(entities))

    def iter_masked_texts(
        self,
        *,
        split: str,
        allow_network: bool = False,
        trust_remote_code: bool = False,
    ) -> Iterator[str]:
        """Stream masked texts from a masked-pair dataset."""

        _require_network(allow_network, dataset_name=self.dataset_name)

        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            self.dataset_name,
            self.config_name,
            split=split,
            trust_remote_code=bool(trust_remote_code),
        )
        columns = set(ds.column_names)

        masked_field = next((c for c in self.masked_field_candidates if c in columns), None)
        if masked_field is None:
            raise KeyError(
                "Unable to infer masked field. "
                f"columns={sorted(columns)} "
                f"masked_candidates={list(self.masked_field_candidates)}"
            )

        for row in ds:
            masked = str(row.get(masked_field, "") or "")
            if not masked:
                continue
            yield masked


def adapter_ai4privacy_pii_masking_300k(*, language: str = "en") -> HuggingFaceMaskedTextAdapter:
    """ai4privacy/pii-masking-300k adapter (masked-pair → gold spans)."""

    return HuggingFaceMaskedTextAdapter(
        "ai4privacy/pii-masking-300k",
        original_field_candidates=("original_text", "source_text", "text", "input"),
        masked_field_candidates=("masked_text", "anonymized_text", "masked", "output", "target", "target_text"),
        language=language,
    )


def adapter_nemotron_pii(*, language: str = "en") -> HuggingFaceMaskedTextAdapter:
    """nvidia/Nemotron-PII adapter (masked-pair → gold spans)."""

    return HuggingFaceMaskedTextAdapter(
        "nvidia/Nemotron-PII",
        original_field_candidates=("original_text", "source_text", "text", "input", "prompt"),
        masked_field_candidates=(
            "masked_text",
            "anonymized_text",
            "masked",
            "output",
            "target",
            "completion",
            "text_tagged",
        ),
        language=language,
    )
