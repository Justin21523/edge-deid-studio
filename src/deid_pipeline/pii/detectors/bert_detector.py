# src/deid_pipeline/pii/detectors/bert_detector.py
"""
BERT NER 偵測器實作
支援 Hugging Face transformers 的 BERT 模型進行 PII 偵測
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig
)

from ..utils.base import PIIDetector, Entity
from ...config import Config


logger = logging.getLogger(__name__)

@dataclass
class TokenInfo:
    """Token 資訊"""
    token: str
    start_pos: int
    end_pos: int
    token_id: int
    is_subword: bool = False

class BertNERDetector(PIIDetector):
    """
    使用 BERT 模型的 NER 偵測器

    支援功能:
    - 多語言 BERT 模型載入
    - BIO 標籤解析和實體聚合
    - 批次處理優化
    - 子詞對齊處理
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "bert-base-chinese-ner",
        confidence_threshold: float = Config.BERT_CONFIDENCE_THRESHOLD,
        max_length: int = Config.BERT_MAX_LENGTH,
        batch_size: int = Config.BERT_BATCH_SIZE,
        device: Optional[str] = None
    ):
        """
        初始化 BERT NER 偵測器

        Args:
            model_path: 本地模型路徑
            model_name: 模型名稱
            confidence_threshold: 信心度閾值
            max_length: 最大序列長度
            batch_size: 批次大小
            device: 運算裝置 (cuda/cpu)
        """
        super().__init__()

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.batch_size = batch_size

        # 設定裝置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"使用裝置: {self.device}")

        # 載入模型
        if Config.USE_STUB:
            logger.warning("使用 STUB 模式，跳過模型載入")
            self.model = None
            self.tokenizer = None
            self.config = None
            self.label_list = self._get_default_labels()
        else:
            self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]):
        """載入 BERT 模型和 tokenizer"""
        try:
            if model_path and Path(model_path).exists():
                logger.info(f"從本地載入模型: {model_path}")
                model_dir = Path(model_path)

                # 載入各組件
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_dir / "tokenizer"
                )
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_dir / "pytorch_model"
                )
                self.config = AutoConfig.from_pretrained(
                    model_dir / "config"
                )

            else:
                logger.info(f"從 HuggingFace 載入模型: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
                self.config = AutoConfig.from_pretrained(self.model_name)

            # 移到指定裝置
            self.model.to(self.device)
            self.model.eval()

            # 獲取標籤列表
            self.label_list = list(self.config.id2label.values())

            logger.info(f"模型載入成功，支援標籤: {self.label_list}")

        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            if not Config.USE_STUB:
                raise

    def _get_default_labels(self) -> List[str]:
        """獲取預設標籤列表 (STUB 模式使用)"""
        return [
            "O",
            "B-PERSON", "I-PERSON",
            "B-ID", "I-ID",
            "B-PHONE", "I-PHONE",
            "B-EMAIL", "I-EMAIL",
            "B-ADDRESS", "I-ADDRESS",
            "B-ORG", "I-ORG",
            "B-DATE", "I-DATE"
        ]

    def detect(self, text: str) -> List[Entity]:
        """
        偵測文本中的 PII 實體

        Args:
            text: 輸入文本

        Returns:
            偵測到的實體列表
        """
        if Config.USE_STUB:
            return self._stub_detect(text)

        if not text.strip():
            return []

        try:
            # 文本預處理
            processed_text = self._preprocess_text(text)

            # 分割長文本
            text_chunks = self._split_text(processed_text)

            all_entities = []

            # 批次處理
            for chunk_start, chunk_text in text_chunks:
                chunk_entities = self._detect_chunk(chunk_text, chunk_start)
                all_entities.extend(chunk_entities)

            # 後處理和合併
            final_entities = self._post_process_entities(all_entities, text)

            logger.debug(f"偵測到 {len(final_entities)} 個實體")
            return final_entities

        except Exception as e:
            logger.error(f"BERT NER 偵測錯誤: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        """文本前處理"""
        # 正規化空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _split_text(self, text: str) -> List[Tuple[int, str]]:
        """
        分割長文本為可處理的片段

        Returns:
            [(chunk_start_pos, chunk_text), ...]
        """
        if len(text) <= self.max_length - 100:  # 保留特殊 token 空間
            return [(0, text)]

        chunks = []
        chunk_size = self.max_length - 150  # 保守預留空間
        overlap = 50  # 重疊部分避免實體被切斷

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))

            # 尋找適當的切分點 (避免切斷詞語)
            if end < len(text):
                # 往回找空格或標點
                for i in range(end, max(start + chunk_size // 2, 0), -1):
                    if text[i] in ' \n\t.,;!?，。；！？':
                        end = i + 1
                        break

            chunk_text = text[start:end]
            chunks.append((start, chunk_text))

            if end >= len(text):
                break

            start = end - overlap

        return chunks

    def _detect_chunk(self, text: str, offset: int = 0) -> List[Entity]:
        """偵測單個文本片段"""
        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # 移到裝置
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0]

        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)

        # 獲取預測結果
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        predicted_labels = predicted_labels[0].cpu().numpy()
        predictions_scores = predictions[0].cpu().numpy()

        # 構建 token 資訊
        token_infos = self._build_token_infos(
            tokens, offset_mapping, predicted_labels, predictions_scores
        )

        # BIO 標籤轉實體
        entities = self._bio_to_entities(token_infos, text, offset)

        return entities

    def _build_token_infos(
        self,
        tokens: List[str],
        offset_mapping: torch.Tensor,
        predicted_labels: np.ndarray,
        prediction_scores: np.ndarray
    ) -> List[TokenInfo]:
        """構建 token 資訊列表"""
        token_infos = []

        for i, (token, (start, end)) in enumerate(zip(tokens, offset_mapping)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            label_id = predicted_labels[i]
            confidence = float(prediction_scores[i, label_id])

            if confidence < self.confidence_threshold:
                continue

            token_info = TokenInfo(
                token=token,
                start_pos=int(start),
                end_pos=int(end),
                token_id=label_id,
                is_subword=token.startswith("##")
            )

            token_infos.append(token_info)

        return token_infos

    def _bio_to_entities(
        self,
        token_infos: List[TokenInfo],
        text: str,
        offset: int = 0
    ) -> List[Entity]:
        """將 BIO 標籤轉換為實體"""
        entities = []
        current_entity = None

        for token_info in token_infos:
            label = self.label_list[token_info.token_id]

            if label == "O":
                # 結束當前實體
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            # 解析 BIO 標籤
            bio_tag, entity_type = label.split("-", 1) if "-" in label else ("O", "")

            if bio_tag == "B":
                # 開始新實體
                if current_entity:
                    entities.append(current_entity)

                current_entity = Entity(
                    text=text[token_info.start_pos:token_info.end_pos],
                    entity_type=self._map_entity_type(entity_type),
                    start=token_info.start_pos + offset,
                    end=token_info.end_pos + offset,
                    confidence=0.8  # 預設信心度
                )

            elif bio_tag == "I" and current_entity:
                # 延續當前實體
                current_entity.text = text[
                    current_entity.start - offset:token_info.end_pos
                ]
                current_entity.end = token_info.end_pos + offset

        # 處理最後一個實體
        if current_entity:
            entities.append(current_entity)

        return entities

    def _map_entity_type(self, bert_type: str) -> str:
        """映射 BERT 標籤到系統實體類型"""
        mapping = {
            "PERSON": "PERSON",
            "ID": "ID_NUMBER",
            "PHONE": "PHONE_NUMBER",
            "EMAIL": "EMAIL_ADDRESS",
            "ADDRESS": "ADDRESS",
            "ORG": "ORGANIZATION",
            "DATE": "DATE"
        }
        return mapping.get(bert_type, bert_type)

    def _post_process_entities(self, entities: List[Entity], original_text: str) -> List[Entity]:
        """後處理實體列表"""
        if not entities:
            return entities

        # 去重和合併重疊實體
        entities = self._remove_duplicates(entities)
        entities = self._merge_adjacent_entities(entities)

        # 驗證實體文本
        validated_entities = []
        for entity in entities:
            if self._validate_entity(entity, original_text):
                validated_entities.append(entity)

        return validated_entities

    def _remove_duplicates(self, entities: List[Entity]) -> List[Entity]:
        """移除重複實體"""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity.start, entity.end, entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _merge_adjacent_entities(self, entities: List[Entity]) -> List[Entity]:
        """合併相鄰的同類型實體"""
        if not entities:
            return entities

        # 按位置排序
        entities.sort(key=lambda x: x.start)

        merged = [entities[0]]

        for current in entities[1:]:
            last = merged[-1]

            # 檢查是否可以合併
            if (current.entity_type == last.entity_type and
                current.start <= last.end + 2):  # 允許小間隔

                # 合併實體
                last.end = max(last.end, current.end)
                last.text = last.text + " " + current.text  # 簡化合併
                last.confidence = max(last.confidence, current.confidence)
            else:
                merged.append(current)

        return merged

    def _validate_entity(self, entity: Entity, original_text: str) -> bool:
        """驗證實體有效性"""
        # 檢查位置範圍
        if entity.start < 0 or entity.end > len(original_text):
            return False

        # 檢查文本一致性
        actual_text = original_text[entity.start:entity.end]
        if not actual_text.strip():
            return False

        # 更新實體文本為實際文本
        entity.text = actual_text

        return True

    def _stub_detect(self, text: str) -> List[Entity]:
        """STUB 模式的簡單偵測 (用於測試)"""
        entities = []

        # 簡單的正則匹配模擬
        patterns = {
            "EMAIL_ADDRESS": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE_NUMBER": r'\b(?:\+?886[-\s]?)?(?:0)?[0-9][-\s]?[0-9]{3,4}[-\s]?[0-9]{3,4}\b',
            "ID_NUMBER": r'\b[A-Z][0-9]{9}\b'
        }

        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(),
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))

        return entities

    def get_supported_entities(self) -> List[str]:
        """獲取支援的實體類型"""
        if Config.USE_STUB:
            return ["EMAIL_ADDRESS", "PHONE_NUMBER", "ID_NUMBER"]

        supported = set()
        for label in self.label_list:
            if label != "O" and "-" in label:
                _, entity_type = label.split("-", 1)
                supported.add(self._map_entity_type(entity_type))

        return list(supported)
