# src/deid_pipeline/pii/detectors/composite.py
"""
複合偵測器實作 - 整合多種 PII 偵測器
支援 BERT PyTorch、BERT ONNX、Regex、spaCy 等偵測器的組合使用
"""

import logging
import time
from typing import List, Dict, Optional, Union, Set
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.base import PIIDetector, Entity
from ...config import Config
from .regex_detector import RegexDetector
from .bert_detector import BertNERDetector
from .bert_onnx_detector import BertONNXNERDetector

logger = logging.getLogger(__name__)

# 條件性導入 spaCy
try:
    from .legacy.spacy_detector import SpacyDetector
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy 不可用，將跳過 SpacyDetector")


@dataclass
class DetectorConfig:
    """偵測器配置"""
    detector_type: str
    enabled: bool = True
    weight: float = 1.0
    config: Dict = field(default_factory=dict)

class CompositeDetector(PIIDetector):
    """
    複合 PII 偵測器

    整合多種偵測器，提供統一的 PII 偵測接口：
    - 自動載入和配置各種偵測器
    - 智能結果合併和衝突解決
    - 效能優化和快取機制
    - 可配置的偵測器組合策略
    """

    def __init__(
        self,
        detector_configs: Optional[List[DetectorConfig]] = None,
        merge_strategy: str = "weighted_vote",
        confidence_threshold: float = 0.5
    ):
        """
        初始化複合偵測器

        Args:
            detector_configs: 偵測器配置列表
            merge_strategy: 合併策略 ("weighted_vote", "max_confidence", "union")
            confidence_threshold: 全域信心度閾值
        """
        super().__init__()

        self.merge_strategy = merge_strategy
        self.confidence_threshold = confidence_threshold
        self.detectors: Dict[str, PIIDetector] = {}
        self.detector_weights: Dict[str, float] = {}

        # 使用預設配置或用戶配置
        if detector_configs is None:
            detector_configs = self._get_default_configs()

        # 初始化偵測器
        self._initialize_detectors(detector_configs)

        logger.info(f"複合偵測器初始化完成，啟用: {list(self.detectors.keys())}")

    def _get_default_configs(self) -> List[DetectorConfig]:
        """獲取預設偵測器配置"""
        configs = [
            DetectorConfig(
                detector_type="regex",
                enabled=True,
                weight=0.8,
                config={"use_zh_rules": True, "use_en_rules": True}
            ),
            DetectorConfig(
                detector_type="bert_onnx",
                enabled=not Config.USE_STUB,
                weight=1.2,
                config={
                    "model_name": "bert-base-chinese-ner",
                    "confidence_threshold": 0.7
                }
            ),
            DetectorConfig(
                detector_type="bert_pytorch",
                enabled=False,  # 預設關閉 PyTorch 版本，優先使用 ONNX
                weight=1.0,
                config={
                    "model_name": "bert-base-chinese-ner",
                    "confidence_threshold": 0.7
                }
            )
        ]

        # 條件性添加 spaCy
        if SPACY_AVAILABLE:
            configs.append(DetectorConfig(
                detector_type="spacy",
                enabled=not Config.USE_STUB,
                weight=0.9,
                config={"model_name": "zh_core_web_sm"}
            ))

        return configs

    def _initialize_detectors(self, configs: List[DetectorConfig]):
        """初始化各個偵測器"""
        for config in configs:
            if not config.enabled:
                continue

            try:
                detector = self._create_detector(config)
                if detector:
                    self.detectors[config.detector_type] = detector
                    self.detector_weights[config.detector_type] = config.weight
                    logger.info(f"成功載入偵測器: {config.detector_type}")

            except Exception as e:
                logger.error(f"載入偵測器 {config.detector_type} 失敗: {e}")
                if not Config.USE_STUB:
                    # 在非 STUB 模式下，某些偵測器載入失敗不應影響整體系統
                    continue

    def _create_detector(self, config: DetectorConfig) -> Optional[PIIDetector]:
        """創建具體的偵測器實例"""
        detector_type = config.detector_type.lower()

        if detector_type == "regex":
            return RegexDetector(**config.config)

        elif detector_type == "bert_onnx":
            return BertONNXNERDetector(**config.config)

        elif detector_type == "bert_pytorch":
            return BertNERDetector(**config.config)

        elif detector_type == "spacy" and SPACY_AVAILABLE:
            return SpacyDetector(**config.config)

        else:
            logger.warning(f"未知偵測器類型: {detector_type}")
            return None

    def detect(self, text: str) -> List[Entity]:
        """
        使用所有啟用的偵測器進行 PII 偵測

        Args:
            text: 輸入文本

        Returns:
            合併後的實體列表
        """
        if not text.strip():
            return []

        # 收集所有偵測器的結果
        all_entities = []
        detector_results = {}

        for detector_name, detector in self.detectors.items():
            try:
                entities = detector.detect(text)
                detector_results[detector_name] = entities

                # 為實體添加來源標記和權重
                for entity in entities:
                    entity.source = detector_name
                    entity.weight = self.detector_weights.get(detector_name, 1.0)

                all_entities.extend(entities)
                logger.debug(f"{detector_name} 偵測到 {len(entities)} 個實體")

            except Exception as e:
                logger.error(f"偵測器 {detector_name} 執行失敗: {e}")
                continue

        if not all_entities:
            return []

        # 合併和去重
        merged_entities = self._merge_entities(all_entities, text)

        # 過濾低信心度實體
        filtered_entities = [
            entity for entity in merged_entities
            if entity.confidence >= self.confidence_threshold
        ]

        logger.debug(f"複合偵測完成: {len(filtered_entities)} 個最終實體")
        return filtered_entities

    def _merge_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        合併來自不同偵測器的實體

        處理流程:
        1. 按位置排序
        2. 識別重疊區域
        3. 根據策略合併重疊實體
        4. 解決衝突
        """
        if not entities:
            return []

        # 按起始位置排序
        entities.sort(key=lambda x: (x.start, x.end))

        merged = []
        i = 0

        while i < len(entities):
            current = entities[i]
            overlapping = [current]

            # 找到所有與當前實體重疊的實體
            j = i + 1
            while j < len(entities) and entities[j].start < current.end:
                overlapping.append(entities[j])
                j += 1

            # 合併重疊實體
            if len(overlapping) == 1:
                merged.append(current)
            else:
                merged_entity = self._merge_overlapping_entities(overlapping, text)
                if merged_entity:
                    merged.append(merged_entity)

            i = j

        return merged

    def _merge_overlapping_entities(self, entities: List[Entity], text: str) -> Optional[Entity]:
        """
        合併重疊的實體組

        Args:
            entities: 重疊的實體列表
            text: 原始文本

        Returns:
            合併後的實體
        """
        if not entities:
            return None

        if len(entities) == 1:
            return entities[0]

        # 根據合併策略處理
        if self.merge_strategy == "weighted_vote":
            return self._weighted_vote_merge(entities, text)
        elif self.merge_strategy == "max_confidence":
            return self._max_confidence_merge(entities, text)
        elif self.merge_strategy == "union":
            return self._union_merge(entities, text)
        else:
            logger.warning(f"未知合併策略: {self.merge_strategy}")
            return self._max_confidence_merge(entities, text)

    def _weighted_vote_merge(self, entities: List[Entity], text: str) -> Entity:
        """加權投票合併策略"""
        # 計算實體類型的加權分數
        type_scores = {}
        total_weight = 0

        for entity in entities:
            weight = getattr(entity, 'weight', 1.0) * entity.confidence
            entity_type = entity.entity_type

            if entity_type not in type_scores:
                type_scores[entity_type] = 0

            type_scores[entity_type] += weight
            total_weight += weight

        # 選擇得分最高的實體類型
        best_type = max(type_scores, key=type_scores.get)

        # 計算合併範圍
        start = min(entity.start for entity in entities)
        end = max(entity.end for entity in entities)

        # 計算平均信心度
        avg_confidence = sum(entity.confidence for entity in entities) / len(entities)

        # 創建合併實體
        merged_entity = Entity(
            text=text[start:end],
            type=best_type,
            start=start,
            end=end,
            score=min(avg_confidence, 0.95)  # 限制最大信心度
        )

        # 添加來源資訊
        sources = [getattr(entity, 'source', 'unknown') for entity in entities]
        merged_entity.sources = list(set(sources))

        return merged_entity

    def _max_confidence_merge(self, entities: List[Entity], text: str) -> Entity:
        """最大信心度合併策略"""
        # 選擇信心度最高的實體
        best_entity = max(entities, key=lambda x: x.confidence)

        # 計算所有實體的範圍
        start = min(entity.start for entity in entities)
        end = max(entity.end for entity in entities)

        # 創建合併實體，保持最佳實體的類型
        merged_entity = Entity(
            text=text[start:end],
            entity_type=best_entity.entity_type,
            start=start,
            end=end,
            confidence=best_entity.confidence
        )

        # 添加來源資訊
        sources = [getattr(entity, 'source', 'unknown') for entity in entities]
        merged_entity.sources = list(set(sources))

        return merged_entity

    def _union_merge(self, entities: List[Entity], text: str) -> Entity:
        """聯集合併策略"""
        # 使用實體優先級決定類型
        priority_entities = sorted(
            entities,
            key=lambda x: Config.ENTITY_PRIORITY.get(x.entity_type, 999)
        )

        best_entity = priority_entities[0]

        # 計算範圍
        start = min(entity.start for entity in entities)
        end = max(entity.end for entity in entities)

        # 創建合併實體
        merged_entity = Entity(
            text=text[start:end],
            entity_type=best_entity.entity_type,
            start=start,
            end=end,
            confidence=max(entity.confidence for entity in entities)
        )

        # 添加來源資訊
        sources = [getattr(entity, 'source', 'unknown') for entity in entities]
        merged_entity.sources = list(set(sources))

        return merged_entity

    def get_supported_entities(self) -> List[str]:
        """獲取所有偵測器支援的實體類型聯集"""
        all_entities = set()

        for detector in self.detectors.values():
            try:
                supported = detector.get_supported_entities()
                all_entities.update(supported)
            except Exception as e:
                logger.debug(f"獲取偵測器支援實體失敗: {e}")

        return sorted(list(all_entities))

    def get_detector_status(self) -> Dict[str, Dict]:
        """獲取各偵測器狀態"""
        status = {}

        for name, detector in self.detectors.items():
            try:
                detector_info = {
                    "enabled": True,
                    "weight": self.detector_weights.get(name, 1.0),
                    "supported_entities": detector.get_supported_entities()
                }

                # 嘗試獲取模型資訊
                if hasattr(detector, 'get_model_info'):
                    detector_info["model_info"] = detector.get_model_info()

                status[name] = detector_info

            except Exception as e:
                status[name] = {
                    "enabled": False,
                    "error": str(e),
                    "weight": self.detector_weights.get(name, 1.0)
                }

        return status

    def enable_detector(self, detector_name: str):
        """啟用指定偵測器"""
        if detector_name in self.detectors:
            logger.info(f"偵測器 {detector_name} 已啟用")
        else:
            logger.warning(f"偵測器 {detector_name} 不存在")

    def disable_detector(self, detector_name: str):
        """停用指定偵測器"""
        if detector_name in self.detectors:
            del self.detectors[detector_name]
            logger.info(f"已停用偵測器: {detector_name}")
        else:
            logger.warning(f"偵測器 {detector_name} 不存在")

    def set_detector_weight(self, detector_name: str, weight: float):
        """設定偵測器權重"""
        if detector_name in self.detector_weights:
            self.detector_weights[detector_name] = weight
            logger.info(f"設定 {detector_name} 權重為 {weight}")
        else:
            logger.warning(f"偵測器 {detector_name} 不存在")

    def benchmark_detectors(self, test_texts: List[str]) -> Dict[str, Dict]:
        """對各偵測器進行效能測試"""
        results = {}

        for name, detector in self.detectors.items():
            try:
                # 預熱
                if test_texts:
                    detector.detect(test_texts[0])

                # 測試
                start_time = time.time()
                for text in test_texts:
                    detector.detect(text)
                end_time = time.time()

                avg_time = (end_time - start_time) / len(test_texts) if test_texts else 0

                results[name] = {
                    "avg_time_per_text": avg_time,
                    "total_time": end_time - start_time,
                    "status": "success"
                }

                # ONNX 偵測器特殊測試
                if hasattr(detector, 'benchmark'):
                    bench_results = detector.benchmark(test_texts)
                    results[name].update(bench_results)

            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }

        return results
