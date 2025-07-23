from typing import List
from pathlib import Path, PurePath
from optimum.onnxruntime import ORTModelForTokenClassification
import onnxruntime as ort
from transformers import AutoTokenizer
import re, time, logging
import numpy as np
from .base import PIIDetector, Entity
from .. import USE_STUB
from .base import Entity
from .base import PII_TYPES
from deid_pipeline.config import Config

logger = logging.getLogger(__name__)

# 實體類型標準化映射
ENTITY_TYPE_MAP = {
    "PER": "NAME",
    "PERSON": "NAME",
    "LOC": "ADDRESS",
    "GPE": "ADDRESS",
    "ORG": "ORGANIZATION",
    "ID": "ID",
    "PHONE": "PHONE",
    "EMAIL": "EMAIL"
}
class BertNERDetector(PIIDetector):
    def __init__(self, model_dir: str | PurePath, provider: str="CPUExecutionProvider"):
        self.config = Config()
        if not self.config.USE_STUB:
            # 動態選擇最佳執行提供者
            providers = ["CPUExecutionProvider"]
            available_providers = ort.get_available_providers()

            if "NPUExecutionProvider" in available_providers:
                providers.insert(0, "NPUExecutionProvider")
            elif "CUDAExecutionProvider" in available_providers:
                providers.insert(0, "CUDAExecutionProvider")

            logger.info(f"使用ONNX執行提供者: {providers}")

            self.model = ORTModelForTokenClassification.from_pretrained(
                model_dir, providers=providers
            )
            self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
            self.model_max_length = self.tok.model_max_length
        else:
            self.model = None
            logger.warning("使用存根模式，BERT檢測器未初始化")

    def detect(self, text: str) -> List[Entity]:
        start_time = time.perf_counter()

        if self.config.USE_STUB or self.model is None:
            return self._stub_detection(text)

        # 處理長文本的滑動窗口
        entities = []
        stride = self.model_max_length // 2
        chunks = [
            (i, text[i:i+self.model_max_length])
            for i in range(0, len(text), stride)
        ]

        for offset, chunk in chunks:
            chunk_entities = self._process_chunk(chunk, offset)
            entities.extend(chunk_entities)

        # 合併重疊實體
        merged_entities = self._merge_entities(entities)

        # 記錄效能
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(f"BERT檢測完成，字數: {len(text)}, 實體數: {len(merged_entities)}, 耗時: {elapsed:.2f}ms")

        return merged_entities

    def _process_chunk(self, text: str, offset: int = 0) -> List[Entity]:
        """處理文本塊並返回實體"""
        inputs = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True
        )

        # 獲取預測結果
        outputs = self.model(**inputs)
        logits = outputs.logits[0].numpy()
        predictions = np.argmax(logits, axis=1)
        scores = np.max(logits, axis=1)

        # 處理實體
        entities = []
        current_entity = None

        for idx, (token_id, pred_idx, score) in enumerate(zip(inputs.input_ids[0], predictions, scores)):
            # 跳過特殊token
            if token_id in [self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id]:
                continue

            # 獲取原始文本位置
            token_start, token_end = inputs.offset_mapping[0][idx]
            if token_start == token_end == 0:  # 特殊token
                continue

            label = self.model.config.id2label[pred_idx]
            score = float(score)

            # 標準化實體類型
            base_label = label.replace("B-", "").replace("I-", "")
            normalized_type = ENTITY_TYPE_MAP.get(base_label, base_label)

            # 處理實體邊界
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "span": [token_start + offset, token_end + offset],
                    "type": normalized_type,
                    "score": score,
                    "source": "bert"
                }
            elif label.startswith("I-") and current_entity and current_entity["type"] == normalized_type:
                current_entity["span"][1] = token_end + offset
                current_entity["score"] = max(current_entity["score"], score)
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """合併重疊的實體"""
        if not entities:
            return []

        # 按起始位置排序
        entities = sorted(entities, key=lambda x: x["span"][0])
        merged = [entities[0]]

        for current in entities[1:]:
            last = merged[-1]

            # 檢查重疊
            if current["span"][0] <= last["span"][1]:
                # 合併條件：相同類型且重疊部分超過50%
                overlap = min(last["span"][1], current["span"][1]) - current["span"][0]
                min_length = min(
                    last["span"][1] - last["span"][0],
                    current["span"][1] - current["span"][0]
                )

                if current["type"] == last["type"] and overlap > min_length * 0.5:
                    merged[-1]["span"][1] = max(last["span"][1], current["span"][1])
                    merged[-1]["score"] = max(last["score"], current["score"])
                    continue

            merged.append(current)

        return merged

    def _stub_detection(self, text: str) -> List[Entity]:
        """存根模式下的簡單檢測"""
        entities = []

        # 台灣身分證
        for m in re.finditer(r"[A-Z][12]\d{8}", text):
            entities.append({
                "span": [m.start(), m.end()],
                "type": "TW_ID",
                "score": 1.0,
                "source": "regex_stub"
            })

        # 台灣手機號
        for m in re.finditer(r"09\d{2}-?\d{3}-?\d{3}", text):
            entities.append({
                "span": [m.start(), m.end()],
                "type": "PHONE",
                "score": 1.0,
                "source": "regex_stub"
            })

        return entities
