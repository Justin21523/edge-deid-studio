from typing import List
from pathlib import Path, PurePath
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer
from .base import PIIDetector, Entity
from .. import USE_STUB

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
            # 假偵測：把數字 / 中文名字 / email 用 regex 回傳
            import re
            ents = []
            for m in re.finditer(r"[A-Z]\d{9}", text):
                ents.append({"span":[m.start(), m.end()], "type":"ID", "score":1.0})
            for m in re.finditer(r"\d{10}", text):
                ents.append({"span":[m.start(), m.end()], "type":"PHONE", "score":1.0})
            return ents
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
