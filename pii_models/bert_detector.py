from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from .base import PIIDetector, Entity

LABELS = ["O", "B-NAME", "I-NAME", "B-ID", "I-ID", "B-PHONE", "I-PHONE",
          "B-ADDRESS", "I-ADDRESS", "B-EMAIL", "I-EMAIL"]  # 依微調實際標籤調整

class BertNERDetector(PIIDetector):
    def __init__(self, model_dir: str | Path, provider: str = "CPUExecutionProvider"):
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = ORTModelForTokenClassification.from_pretrained(
            model_dir, provider=provider
        )
        self.id2label = {i: lab for i, lab in enumerate(LABELS)}

    @torch.no_grad()
    def detect(self, text: str) -> list[Entity]:
        inputs = self.tok(text, return_tensors="pt")
        out = self.model(**inputs)
        logits = out.logits.cpu().numpy()[0]          # [seq_len, num_labels]
        ids = np.argmax(logits, axis=-1)              # 每個 token 的 label id
        scores = np.max(torch.softmax(out.logits, -1).cpu().numpy()[0], axis=-1)

        entities = []
        current = None
        for i, (lab_id, score) in enumerate(zip(ids, scores)):
            label = self.id2label[lab_id]
            if label.startswith("B-"):
                if current:  # 先收尾上一段
                    entities.append(current)
                start = inputs.token_to_chars(0, i).start
                current = {"span": [start, start], "type": label[2:], "score": float(score)}
            elif label.startswith("I-") and current:
                current["span"][1] = inputs.token_to_chars(0, i).end
                current["score"] = max(current["score"], float(score))
            else:
                if current:
                    entities.append(current)
                    current = None
        if current:
            entities.append(current)
        return entities
