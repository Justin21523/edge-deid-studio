# src/deid_pipeline/pii/utils/base.py
"""
EdgeDeID Studio - 文字替換處理器
負責根據偵測到的 PII 實體進行遮蔽或假資料替換
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .base import Entity
from .fake_provider import FakeProvider

logger = logging.getLogger(__name__)

class ReplacementMode(Enum):
    """替換模式枚舉"""
    MASK = "mask"          # 遮蔽模式 (黑條)
    REDACT = "redact"      # 隱碼模式 (****)
    FAKE = "fake"          # 假資料替換
    PARTIAL = "partial"    # 部分遮蔽 (保留部分字符)

@dataclass
class ReplacementResult:
    """替換結果"""
    original_text: str
    processed_text: str
    entities_replaced: List[Entity]
    replacement_map: Dict[str, str]  # 原始值 -> 替換值的映射
    statistics: Dict[str, int]       # 統計信息

class Replacer:
    """
    PII 替換處理器

    核心功能:
    1. 根據實體類型選擇替換策略
    2. 支援多種替換模式 (遮蔽/假資料)
    3. 維持文本格式和版面結構
    4. 提供一致性替換 (相同值使用相同假資料)
    """

    def __init__(self,
                 mode: ReplacementMode = ReplacementMode.MASK,
                 fake_provider: FakeProvider = None,
                 preserve_format: bool = True,
                 consistency_cache: bool = True):
        """
        初始化替換器

        Args:
            mode: 替換模式
            fake_provider: 假資料提供器
            preserve_format: 是否保持原始格式
            consistency_cache: 是否啟用一致性快取
        """
        self.mode = mode
        self.fake_provider = fake_provider or FakeProvider()
        self.preserve_format = preserve_format
        self.consistency_cache = consistency_cache

        # 一致性快取 - 確保相同原始值得到相同替換值
        self._replacement_cache: Dict[str, str] = {}

        # 統計信息
        self._stats = {
            'total_entities': 0,
            'replaced_entities': 0,
            'cached_replacements': 0,
            'errors': 0
        }

        # 實體類型特定的替換配置
        self._entity_configs = {
            'PERSON': {
                'mask_char': '█',
                'partial_keep': 1,  # 保留幾個字符
                'fake_type': 'person_name'
            },
            'PHONE': {
                'mask_char': '*',
                'partial_keep': 3,
                'fake_type': 'phone_number',
                'format_pattern': r'(\d{2,3})-?(\d{3,4})-?(\d{4})'
            },
            'EMAIL': {
                'mask_char': '*',
                'partial_keep': 2,
                'fake_type': 'email',
                'format_pattern': r'([^@]+)@([^.]+\..+)'
            },
            'ID_CARD': {
                'mask_char': '*',
                'partial_keep': 2,
                'fake_type': 'id_card',
                'format_pattern': r'([A-Z]\d{9})'
            },
            'ADDRESS': {
                'mask_char': '█',
                'partial_keep': 2,
                'fake_type': 'address'
            },
            'ORGANIZATION': {
                'mask_char': '█',
                'partial_keep': 1,
                'fake_type': 'company'
            },
            'DATE': {
                'mask_char': '*',
                'partial_keep': 0,
                'fake_type': 'date',
                'format_pattern': r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})'
            }
        }

    def replace_entities(self,
                        text: str,
                        entities: List[Entity],
                        mode: Optional[ReplacementMode] = None) -> ReplacementResult:
        """
        替換文本中的 PII 實體

        Args:
            text: 原始文本
            entities: 要替換的實體列表
            mode: 替換模式 (覆蓋預設)

        Returns:
            ReplacementResult: 替換結果
        """
        if not entities:
            return ReplacementResult(
                original_text=text,
                processed_text=text,
                entities_replaced=[],
                replacement_map={},
                statistics=self._stats.copy()
            )

        replacement_mode = mode or self.mode
        processed_text = text
        replacement_map = {}
        successfully_replaced = []

        # 按照實體在文本中的位置倒序排列，避免位置偏移
        sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

        self._stats['total_entities'] += len(entities)

        for entity in sorted_entities:
            try:
                original_value = text[entity.start:entity.end]

                # 選擇替換策略
                if replacement_mode == ReplacementMode.MASK:
                    replacement_value = self._create_mask(entity, original_value)
                elif replacement_mode == ReplacementMode.REDACT:
                    replacement_value = self._create_redaction(entity, original_value)
                elif replacement_mode == ReplacementMode.FAKE:
                    replacement_value = self._create_fake_replacement(entity, original_value)
                elif replacement_mode == ReplacementMode.PARTIAL:
                    replacement_value = self._create_partial_mask(entity, original_value)
                else:
                    logger.warning(f"未知替換模式: {replacement_mode}")
                    continue

                # 執行替換
                processed_text = (
                    processed_text[:entity.start] +
                    replacement_value +
                    processed_text[entity.end:]
                )

                replacement_map[original_value] = replacement_value
                successfully_replaced.append(entity)
                self._stats['replaced_entities'] += 1

                logger.debug(f"替換 {entity.entity_type}: '{original_value}' -> '{replacement_value}'")

            except Exception as e:
                logger.error(f"替換實體時發生錯誤 {entity}: {e}")
                self._stats['errors'] += 1
                continue

        return ReplacementResult(
            original_text=text,
            processed_text=processed_text,
            entities_replaced=successfully_replaced,
            replacement_map=replacement_map,
            statistics=self._stats.copy()
        )

    def _create_mask(self, entity: Entity, original_value: str) -> str:
        """創建遮蔽替換"""
        config = self._entity_configs.get(entity.entity_type, {})
        mask_char = config.get('mask_char', '█')

        if self.preserve_format:
            # 保持原始格式 - 字母/數字用遮蔽字符，其他保持
            result = ''
            for char in original_value:
                if char.isalnum():
                    result += mask_char
                else:
                    result += char
            return result
        else:
            return mask_char * len(original_value)

    def _create_redaction(self, entity: Entity, original_value: str) -> str:
        """創建隱碼替換"""
        config = self._entity_configs.get(entity.entity_type, {})
        redact_length = max(4, min(len(original_value), 8))
        return '*' * redact_length

    def _create_partial_mask(self, entity: Entity, original_value: str) -> str:
        """創建部分遮蔽"""
        config = self._entity_configs.get(entity.entity_type, {})
        keep_chars = config.get('partial_keep', 1)
        mask_char = config.get('mask_char', '*')

        if len(original_value) <= keep_chars * 2:
            return mask_char * len(original_value)

        # 保留前後部分字符
        prefix = original_value[:keep_chars]
        suffix = original_value[-keep_chars:] if keep_chars > 0 else ''
        middle_length = len(original_value) - len(prefix) - len(suffix)

        return prefix + mask_char * middle_length + suffix

    def _create_fake_replacement(self, entity: Entity, original_value: str) -> str:
        """創建假資料替換"""
        # 檢查一致性快取
        cache_key = f"{entity.entity_type}:{original_value}"
        if self.consistency_cache and cache_key in self._replacement_cache:
            self._stats['cached_replacements'] += 1
            return self._replacement_cache[cache_key]

        config = self._entity_configs.get(entity.entity_type, {})
        fake_type = config.get('fake_type', 'text')

        try:
            # 根據實體類型生成假資料
            if entity.entity_type == 'PERSON':
                fake_value = self.fake_provider.generate_fake_name(
                    original_value, preserve_format=self.preserve_format
                )
            elif entity.entity_type == 'PHONE':
                fake_value = self.fake_provider.generate_fake_phone(
                    original_value, preserve_format=self.preserve_format
                )
            elif entity.entity_type == 'EMAIL':
                fake_value = self.fake_provider.generate_fake_email(
                    original_value, preserve_format=self.preserve_format
                )
            elif entity.entity_type == 'ID_CARD':
                fake_value = self.fake_provider.generate_fake_id_card(
                    original_value, preserve_format=self.preserve_format
                )
            elif entity.entity_type == 'ADDRESS':
                fake_value = self.fake_provider.generate_fake_address(
                    original_value, preserve_format=self.preserve_format
                )
            elif entity.entity_type == 'ORGANIZATION':
                fake_value = self.fake_provider.generate_fake_company(
                    original_value, preserve_format=self.preserve_format
                )
            elif entity.entity_type == 'DATE':
                fake_value = self.fake_provider.generate_fake_date(
                    original_value, preserve_format=self.preserve_format
                )
            else:
                # 通用文字生成
                fake_value = self.fake_provider.generate_fake_text(
                    original_value, preserve_format=self.preserve_format
                )

            # 存入快取
            if self.consistency_cache:
                self._replacement_cache[cache_key] = fake_value

            return fake_value

        except Exception as e:
            logger.error(f"生成假資料失敗 {entity.entity_type}: {e}")
            # 降級到遮蔽模式
            return self._create_mask(entity, original_value)

    def _preserve_original_format(self, original: str, replacement: str) -> str:
        """保持原始格式"""
        if not self.preserve_format:
            return replacement

        # 保持大小寫格式
        result = []
        replacement_chars = list(replacement)

        for i, orig_char in enumerate(original):
            if i < len(replacement_chars):
                repl_char = replacement_chars[i]
                if orig_char.isupper():
                    result.append(repl_char.upper())
                elif orig_char.islower():
                    result.append(repl_char.lower())
                else:
                    result.append(repl_char)
            else:
                break

        return ''.join(result)

    def batch_replace(self,
                     text_entities_pairs: List[Tuple[str, List[Entity]]],
                     mode: Optional[ReplacementMode] = None) -> List[ReplacementResult]:
        """
        批次替換多個文本

        Args:
            text_entities_pairs: (文本, 實體列表) 配對
            mode: 替換模式

        Returns:
            List[ReplacementResult]: 替換結果列表
        """
        results = []

        for text, entities in text_entities_pairs:
            result = self.replace_entities(text, entities, mode)
            results.append(result)

        return results

    def get_statistics(self) -> Dict[str, int]:
        """獲取替換統計信息"""
        return self._stats.copy()

    def reset_statistics(self):
        """重置統計信息"""
        self._stats = {
            'total_entities': 0,
            'replaced_entities': 0,
            'cached_replacements': 0,
            'errors': 0
        }

    def clear_cache(self):
        """清空一致性快取"""
        self._replacement_cache.clear()
        logger.info("替換快取已清空")

    def export_replacement_map(self) -> Dict[str, str]:
        """匯出替換映射表"""
        return self._replacement_cache.copy()

    def import_replacement_map(self, replacement_map: Dict[str, str]):
        """匯入替換映射表"""
        self._replacement_cache.update(replacement_map)
        logger.info(f"已匯入 {len(replacement_map)} 個替換映射")

    def set_entity_config(self, entity_type: str, config: Dict):
        """設定特定實體類型的替換配置"""
        self._entity_configs[entity_type] = config
        logger.info(f"已更新實體類型 {entity_type} 的替換配置")

    def validate_replacement(self, original: str, replacement: str) -> bool:
        """驗證替換結果是否合理"""
        # 基本驗證規則
        if not replacement or len(replacement.strip()) == 0:
            return False

        # 長度檢查 - 替換值不應該比原始值長太多
        if len(replacement) > len(original) * 3:
            return False

        # 避免意外洩露原始信息
        if original.lower() in replacement.lower():
            return False

        return True
