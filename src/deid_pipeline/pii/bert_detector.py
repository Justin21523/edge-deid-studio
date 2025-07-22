from typing import List
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer
from .base import PIIDetector, Entity

class BertNERDetector(PIIDetector):
    def __init__(self, model_dir: str, provider: str="CPUExecutionProvider"):
        self.model = ORTModelForTokenClassification.from_pretrained(model_dir, provider=provider)
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    def detect(self, text: str) -> List[Entity]:
        inputs = self.tok(text, return_tensors="pt")
        outputs = self.model(**inputs).logits[0]  # (seq_len, num_labels)
        spans = []
        for idx, logit in enumerate(outputs):
            label_id = logit.argmax().item()
            score = float(logit[label_id])
            label = self.model.config.id2label[label_id]
            if score > 0.85 and label != "O":
                start, end = inputs.token_to_chars(idx)
                spans.append({"span":[start, end],"type":label,"score":score})
        return sorted(spans, key=lambda x: x["span"][0])
