from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List

import torch

from ...config import Config
from ...pii.utils import logger
from ...runtime.registry import ensure_local_path, get_hf_token_classifier, get_hf_tokenizer
from ..utils.base import Entity, PIIDetector


ENTITY_TYPE_MAP = {
    "PER": "NAME",
    "PERSON": "NAME",
    "LOC": "ADDRESS",
    "GPE": "ADDRESS",
    "ORG": "ORGANIZATION",
    "ID": "ID",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL",
}


class BertNERDetector(PIIDetector):
    """Hugging Face Transformers token-classification detector with caching."""

    def __init__(self, model_dir: str | Path, provider: str = "CPUExecutionProvider"):
        self.config = Config()
        self.model = None
        self.tok = None
        self.model_max_length = 512

        if self.config.USE_STUB:
            logger.info("BERT detector is disabled (USE_STUB=true).")
            return

        model_path = ensure_local_path(Path(model_dir))
        try:
            self.model = get_hf_token_classifier(model_path)
            self.tok = get_hf_tokenizer(model_path)
            self.model.eval()
            self.model_max_length = int(getattr(self.tok, "model_max_length", 512) or 512)
        except Exception as exc:
            logger.warning("Failed to initialize HF detector; falling back to stub: %s", exc)
            self.model = None
            self.tok = None

    def detect(self, text: str) -> List[Entity]:
        start_time = time.perf_counter()

        if self.config.USE_STUB or self.model is None or self.tok is None:
            return self._stub_detection(text)

        entities: List[Entity] = []
        stride = max(1, self.model_max_length // 2)
        chunks = [(i, text[i : i + self.model_max_length]) for i in range(0, len(text), stride)]

        for offset, chunk in chunks:
            entities.extend(self._process_chunk(chunk, offset))

        merged_entities = self._merge_entities(entities)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug(
            "BERT detection completed (chars=%d, entities=%d, time_ms=%.2f)",
            len(text),
            len(merged_entities),
            elapsed_ms,
        )

        return merged_entities

    def _process_chunk(self, text: str, offset: int = 0) -> List[Entity]:
        inputs = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
        )

        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in inputs.items() if k != "offset_mapping"})

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        pred_conf = torch.max(probs, dim=-1).values.tolist()

        entities: List[Entity] = []
        current_entity = None

        for idx, (token_id, label_id, confidence) in enumerate(
            zip(inputs.input_ids[0].tolist(), pred_ids, pred_conf)
        ):
            if token_id in {self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id}:
                continue

            token_start, token_end = inputs.offset_mapping[0][idx]
            if token_start == token_end == 0:
                continue

            label = self.model.config.id2label[label_id]
            confidence = float(confidence)

            base_label = label.replace("B-", "").replace("I-", "")
            normalized_type = ENTITY_TYPE_MAP.get(base_label, base_label)

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "span": [int(token_start) + offset, int(token_end) + offset],
                    "type": normalized_type,
                    "score": confidence,
                    "source": "bert",
                }
            elif (
                label.startswith("I-")
                and current_entity
                and current_entity["type"] == normalized_type
            ):
                current_entity["span"][1] = int(token_end) + offset
                current_entity["score"] = max(float(current_entity["score"]), confidence)
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        if not entities:
            return []

        entities = sorted(entities, key=lambda x: x["span"][0])
        merged: List[Entity] = [entities[0]]

        for current in entities[1:]:
            last = merged[-1]
            if current["span"][0] <= last["span"][1]:
                overlap = min(last["span"][1], current["span"][1]) - current["span"][0]
                min_length = min(
                    last["span"][1] - last["span"][0],
                    current["span"][1] - current["span"][0],
                )
                if current["type"] == last["type"] and overlap > min_length * 0.5:
                    merged[-1]["span"][1] = max(last["span"][1], current["span"][1])
                    merged[-1]["score"] = max(last["score"], current["score"])
                    continue

            merged.append(current)

        return merged

    def _stub_detection(self, text: str) -> List[Entity]:
        """Simple regex-based stub used when models are disabled/unavailable."""

        entities: List[Entity] = []

        for match in re.finditer(r"[A-Z][12]\d{8}", text):
            entities.append(
                {"span": [match.start(), match.end()], "type": "ID", "score": 1.0, "source": "regex_stub"}
            )

        for match in re.finditer(r"09\d{2}-?\d{3}-?\d{3}", text):
            entities.append(
                {"span": [match.start(), match.end()], "type": "PHONE", "score": 1.0, "source": "regex_stub"}
            )

        return entities
