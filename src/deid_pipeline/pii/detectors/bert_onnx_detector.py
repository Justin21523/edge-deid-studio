from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ...config import Config
from ...pii.utils import logger
from ...runtime.registry import ensure_local_path, get_hf_tokenizer, get_onnx_session
from ...runtime.onnx import select_onnx_providers
from ..utils.base import Entity, PIIDetector
from .bert_detector import ENTITY_TYPE_MAP


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class BertONNXNERDetector(PIIDetector):
    """ONNX Runtime token-classification detector with cached session and tokenizer."""

    def __init__(
        self,
        onnx_model_path: Path,
        tokenizer_dir: Path,
        *,
        providers: List[str] | Tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    ) -> None:
        self.config = Config()

        if self.config.USE_STUB:
            raise RuntimeError("ONNX detector is disabled (USE_STUB=true).")

        onnx_path = Path(onnx_model_path).expanduser().resolve()
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        model_dir = ensure_local_path(Path(tokenizer_dir))
        self.tokenizer = get_hf_tokenizer(model_dir)

        try:
            from transformers import AutoConfig  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("transformers is required for ONNX NER config loading") from exc

        cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        id2label_raw: Dict[int | str, str] = getattr(cfg, "id2label", {}) or {}
        self.id2label = {int(k): str(v) for k, v in id2label_raw.items()}

        self.max_len = int(getattr(self.config, "MAX_SEQ_LENGTH", 512) or 512)
        overlap = int(self.max_len * float(getattr(self.config, "WINDOW_STRIDE", 0.5)))
        self.stride = max(0, min(self.max_len - 1, overlap))

        resolved_providers = select_onnx_providers(providers)
        self.session = get_onnx_session(str(onnx_path), providers=resolved_providers)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, text: str) -> List[Entity]:
        start_time = time.perf_counter()

        encoding = self.tokenizer(
            text,
            return_tensors="np",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding="max_length",
        )

        ort_inputs = {k: encoding[k] for k in self.input_names if k in encoding}
        logits = self.session.run([self.output_name], ort_inputs)[0]
        probs = _softmax(logits)
        pred_ids = probs.argmax(axis=-1)
        pred_conf = probs.max(axis=-1)

        entities: List[Entity] = []
        threshold = float(self.config.BERT_CONFIDENCE_THRESHOLD)
        ignore_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}

        offsets = encoding["offset_mapping"]
        input_ids = encoding["input_ids"]

        for chunk_idx in range(int(input_ids.shape[0])):
            current: Entity | None = None
            for token_id, label_id, conf, (tok_start, tok_end) in zip(
                input_ids[chunk_idx].tolist(),
                pred_ids[chunk_idx].tolist(),
                pred_conf[chunk_idx].tolist(),
                offsets[chunk_idx].tolist(),
            ):
                if token_id in ignore_ids:
                    continue
                if tok_start == tok_end == 0:
                    continue

                label = self.id2label.get(int(label_id), "O")
                confidence = float(conf)
                if label == "O" or confidence < threshold:
                    if current is not None:
                        entities.append(current)
                        current = None
                    continue

                base = label.replace("B-", "").replace("I-", "")
                entity_type = ENTITY_TYPE_MAP.get(base, base)

                if label.startswith("B-"):
                    if current is not None:
                        entities.append(current)
                    current = {
                        "span": [int(tok_start), int(tok_end)],
                        "type": str(entity_type),
                        "score": confidence,
                        "source": "onnx",
                    }
                elif (
                    label.startswith("I-")
                    and current is not None
                    and current.get("type") == str(entity_type)
                ):
                    current["span"][1] = int(tok_end)
                    current["score"] = max(float(current["score"]), confidence)
                else:
                    if current is not None:
                        entities.append(current)
                    current = None

            if current is not None:
                entities.append(current)

        merged_entities = self._merge_entities(entities)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug(
            "ONNX detection completed (chars=%d, chunks=%d, entities=%d, time_ms=%.2f)",
            len(text),
            int(input_ids.shape[0]),
            len(merged_entities),
            elapsed_ms,
        )

        return merged_entities

    @staticmethod
    def _merge_entities(entities: List[Entity]) -> List[Entity]:
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
