# src/deid_pipeline/pii/detectors/bert_onnx_detector.py
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification

from deid_pipeline.config import Config
from deid_pipeline.pii.utils.base import PIIDetector, Entity

class BertONNXNERDetector(PIIDetector):
    def __init__(self):
        # 載入模型（ONNX or PyTorch）
        if Config.USE_ONNX:
            self.model = ORTModelForTokenClassification.from_pretrained(
                Config.ONNX_MODEL_PATH, providers=Config.ONNX_PROVIDERS
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(Config.NER_MODEL_PATH)
            )
        # 共享 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(Config.NER_MODEL_PATH))
        self.id2label  = self.model.config.id2label
        self.max_len   = Config.MAX_SEQ_LENGTH
        self.stride    = int(self.max_len * Config.WINDOW_STRIDE)

    def detect(self, text: str) -> List[Entity]:
        entities = []
        # 使用 sliding window 處理超長文本
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
            stride=self.stride,
            return_overflowing_tokens=True
        )
        input_ids = encoding["input_ids"]
        offsets   = encoding["offset_mapping"]

        for i in range(input_ids.shape[0]):
            # 拆出第 i 片段
            chunk_ids = input_ids[i : i+1]
            chunk_off = offsets[i].tolist()

            if Config.USE_ONNX:
                ort_inputs = {k: v.cpu().numpy() for k, v in encoding.items() if k.startswith("input")}
                logits = self.model.run(None, ort_inputs)[0]  # shape (1, seq_len, n_labels)
                logits = torch.from_numpy(logits)
            else:
                outputs = self.model(**{"input_ids": chunk_ids})
                logits  = outputs.logits  # torch.Tensor (1, seq_len, n_labels)

            scores = torch.softmax(logits, dim=-1)[0]
            preds  = torch.argmax(scores, dim=-1).tolist()
            confid = torch.max(scores, dim=-1).values.tolist()

            for idx, label_id in enumerate(preds):
                label = self.id2label[label_id]
                if label != "O" and confid[idx] >= Config.BERT_CONFIDENCE_THRESHOLD:
                    start, end = chunk_off[idx]
                    entities.append(Entity(
                        span=(start, end),
                        type=label,
                        score=float(confid[idx]),
                        source="bert"
                    ))
        return entities
