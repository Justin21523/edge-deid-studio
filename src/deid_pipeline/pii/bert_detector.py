from typing import List
from pathlib import Path, PurePath
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer
from .base import PIIDetector, Entity
from .. import USE_STUB
import time, logging
from .base import Entity
from .base import PII_TYPES

# 新增類型映射
ENTITY_TYPE_MAP = {
    "PER": "NAME",
    "LOC": "ADDRESS",
    "ORG": "ORGANIZATION",
    "ID": "ID",
    "PHONE": "PHONE"
}

class BertNERDetector(PIIDetector):
    def __init__(self, model_dir: str | PurePath, provider: str="CPUExecutionProvider"):
        if USE_STUB:
            self.stub = True
            return
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"BERT NER ONNX 路徑不存在：{model_dir}\n"
                "請先執行 scripts/export_bert_onnx.py 產生模型"
            )
        self.model = ORTModelForTokenClassification.from_pretrained(model_dir, provider=provider)
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    def detect(self, text: str) -> List[Entity]:
        if getattr(self, "stub", False):
            # stub logic...
            return self._stub_detect(text)

        max_len = self.tok.model_max_length
        stride = max_len // 2
        all_entities: List[Entity] = []

        for start in range(0, len(text), stride):
            chunk = text[start : start + max_len]
            inputs = self.tok(chunk, return_tensors="pt", truncation=True)
            logits = self.model(**inputs).logits[0]  # (seq_len, num_labels)
            ents = self._process_logits(inputs, logits, offset=start)
            all_entities.extend(ents)

        # 合併跨 chunk 重疊與重複
        return self._merge_entities(all_entities)

    def _process_logits(self, inputs, logits, offset=0) -> List[Entity]:
        entities = []
        current = None
        for idx, scores in enumerate(logits):
            token = inputs.tokens()[idx]

            if token in ("[CLS]","[SEP]"): # 跳過特殊token
                continue
            label_id = int(scores.argmax())
            label = self.model.config.id2label[label_id]
            score = float(scores[label_id])
            # 處理B-和I-標籤
            if label.startswith("B-"):
                if current:
                    entities.append(current)
                base = label.split("-")[1]
                current = {
                    "span": [ inputs.token_to_chars(idx).start + offset,
                              inputs.token_to_chars(idx).end   + offset ],
                    "type": ENTITY_TYPE_MAP.get(base, base),
                    "score": score,
                    "source": "bert"
                }
            elif label.startswith("I-") and current:
                end = inputs.token_to_chars(idx).end + offset
                current["span"][1] = end
                current["score"] = max(current["score"], score)
            else:
                if current:
                    entities.append(current)
                    current = None
        if current:
            entities.append(current)
        return entities

    def _merge_entities(self, ents: List[Entity]) -> List[Entity]:
        # 去重 & 衝突解決：保最高 score
        ents = sorted(ents, key=lambda e: e["span"][0])
        resolved = []
        for e in ents:
            if not resolved or e["span"][0] >= resolved[-1]["span"][1]:
                resolved.append(e)
            else:
                if e["score"] > resolved[-1]["score"]:
                    resolved[-1] = e
        return resolved
