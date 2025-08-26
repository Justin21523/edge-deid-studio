# src/deid_pipeline/pii/detectors/__init__.py
from pathlib import Path
import os
from .bert_detector import BertNERDetector
from .regex_detector import RegexDetector
from .bert_onnx_detector import BertONNXNERDetector
from .composite import CompositeDetector
from .legacy.spacy_detector import SpacyDetector
from ...config import Config
from ..utils import logger

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
MODEL_ZH = Config.BERT_MODEL_PATH  # use central config
MODEL_EN = Path(os.getenv("NER_MODEL_PATH_EN", str(PROJECT_ROOT/"models"/"bert-ner-en")))

def get_detector(lang: str = "zh") -> CompositeDetector:
    cfg = Config()
    # 選擇主偵測器：ONNX > HF-BERT > spaCy
    use_onnx = cfg.USE_ONNX and cfg.ONNX_MODEL_PATH.exists()
    use_bert = not cfg.USE_STUB and cfg.BERT_MODEL_PATH.exists()

    bert_cls = BertONNXNERDetector if use_onnx else BertNERDetector
    bert_path = (str(cfg.ONNX_MODEL_PATH) if use_onnx else str(cfg.BERT_MODEL_PATH))

    detectors = []
    # 先嘗試 BERT／ONNX
    if use_bert:
        logger.info(f"使用 {'ONNX' if use_onnx else 'HF-BERT'} NER ({lang})")
        detectors.append(bert_cls(bert_path))

    # Regex 始終作為補漏
    regex_path = cfg.REGEX_RULES_FILE if lang == "zh" else cfg.REGEX_EN_RULES_FILE
    detectors.append(RegexDetector(regex_path))

    # 如果前面都沒加到主偵測器，就 fallback spaCy + Regex
    if not detectors or cfg.USE_STUB:
        logger.info(f"使用 spaCy 偵測 (備用方案 {lang})")
        detectors = [SpacyDetector(), RegexDetector(regex_path)]

    return CompositeDetector(*detectors)

#!/usr/bin/env python3
"""
圖像處理模組串接實作
整合 OCR、文字抽取與 BERT NER 偵測器的完整流程
"""

import logging
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np

from ..parser.text_extractor import SmartTextExtractor
from ..parser.ocr import OCRAdapter
from ..parser.position_mapper import TextPositionMapper
from ..parser.layout import DocumentLayout, PageLayout, TextBlock
from .composite import CompositeDetector
from ..utils.replacer import Replacer
from ..utils.fake_provider import FakeProvider
from ..base import Entity
from ..config import USE_STUB

logger = logging.getLogger(__name__)

class ImageDeidProcessor:
    """
    圖像去識別化處理器

    整合流程:
    1. 圖像預處理 (OCRAdapter)
    2. 文字抽取 (SmartTextExtractor)
    3. PII 偵測 (CompositeDetector with BERT)
    4. 座標映射 (TextPositionMapper)
    5. 去識別化替換 (Replacer + FakeProvider)
    6. 可視化標註
    """

    def __init__(
        self,
        use_onnx: bool = True,
        ocr_engine: str = "auto",
        replacement_mode: str = "mask",  # "mask" or "fake"
        debug_mode: bool = False
    ):
        """
        初始化圖像去識別化處理器

        Args:
            use_onnx: 是否使用 ONNX BERT 模型
            ocr_engine: OCR 引擎選擇
            replacement_mode: 替換模式 (遮蔽/假資料)
            debug_mode: 除錯模式，輸出中間結果
        """
        self.debug_mode = debug_mode
        self.replacement_mode = replacement_mode

        # 初始化核心組件
        self._initialize_components(use_onnx, ocr_engine)

        logger.info("圖像去識別化處理器初始化完成")

    def _initialize_components(self, use_onnx: bool, ocr_engine: str):
        """初始化各個組件"""
        try:
            # 1. 文字抽取器 (包含 OCR)
            self.text_extractor = SmartTextExtractor(
                ocr_engine=ocr_engine,
                enable_layout_analysis=True
            )

            # 2. PII 偵測器 (優先使用 BERT)
            detector_configs = self._get_detector_configs(use_onnx)
            self.pii_detector = CompositeDetector(
                detector_configs=detector_configs,
                merge_strategy="weighted_vote",
                confidence_threshold=0.6
            )

            # 3. 位置映射器
            self.position_mapper = TextPositionMapper()

            # 4. 替換器和假資料生成器
            self.fake_provider = FakeProvider()
            self.replacer = Replacer(fake_provider=self.fake_provider)

            if self.debug_mode:
                self._log_component_status()

        except Exception as e:
            logger.error(f"組件初始化失敗: {e}")
            if not USE_STUB:
                raise

    def _get_detector_configs(self, use_onnx: bool):
        """獲取偵測器配置"""
        from .composite import DetectorConfig

        configs = [
            # Regex 偵測器 - 基礎保證
            DetectorConfig(
                detector_type="regex",
                enabled=True,
                weight=0.7,
                config={"use_zh_rules": True, "use_en_rules": True}
            )
        ]

        # BERT 偵測器 - 主力
        if use_onnx:
            configs.append(DetectorConfig(
                detector_type="bert_onnx",
                enabled=not USE_STUB,
                weight=1.3,
                config={
                    "model_name": "bert-base-chinese-ner",
                    "confidence_threshold": 0.65,
                    "max_length": 512
                }
            ))
        else:
            configs.append(DetectorConfig(
                detector_type="bert_pytorch",
                enabled=not USE_STUB,
                weight=1.2,
                config={
                    "model_name": "bert-base-chinese-ner",
                    "confidence_threshold": 0.65
                }
            ))

        return configs

    def process_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        處理單張圖像的完整去識別化流程

        Args:
            image_path: 圖像路徑或 numpy 陣列
            output_path: 輸出路徑

        Returns:
            處理結果字典
        """
        try:
            # 1. 載入和預處理圖像
            if isinstance(image_path, np.ndarray):
                image = image_path
                image_name = "input_image"
            else:
                image_path = Path(image_path)
                image = cv2.imread(str(image_path))
                image_name = image_path.name

            if image is None:
                raise ValueError("無法載入圖像")

            logger.info(f"開始處理圖像: {image_name}")

            # 2. 抽取文字和版面資訊
            extraction_result = self.text_extractor.extract_from_image(image)

            if not extraction_result.get("text"):
                logger.warning("圖像中未抽取到文字")
                return self._create_empty_result(image, image_name)

            text = extraction_result["text"]
            layout = extraction_result.get("layout")

            if self.debug_mode:
                logger.debug(f"抽取文字長度: {len(text)}")
                logger.debug(f"版面區塊數: {len(layout.pages[0].blocks) if layout else 0}")

            # 3. PII 偵測
            entities = self.pii_detector.detect(text)

            if not entities:
                logger.info("未偵測到 PII 實體")
                return self._create_empty_result(image, image_name)

            logger.info(f"偵測到 {len(entities)} 個 PII 實體")

            # 4. 座標映射
            mapped_entities = self._map_entities_to_image(entities, layout, image.shape)

            # 5. 執行去識別化
            processed_image, processed_text = self._apply_deidentification(
                image, text, mapped_entities
            )

            # 6. 生成結果
            result = self._create_result(
                original_image=image,
                processed_image=processed_image,
                original_text=text,
                processed_text=processed_text, 
                entities=mapped_entities,
                image_name=image_name
            )

            # 7. 保存結果
            if output_path:
                self._save_results(result, output_path)

            return result

        except Exception as e:
            logger.error(f"圖像處理失敗: {e}")
            return {"error": str(e), "success": False}

    def _map_entities_to_image(
        self,
        entities: List[Entity],
        layout: Optional[DocumentLayout],
        image_shape: Tuple[int, int, int]
    ) -> List[Entity]:
        """將文字實體映射到圖像座標"""
        if not layout or not layout.pages:
            logger.warning("無版面資訊，無法進行座標映射")
            return entities

        page_layout = layout.pages[0]  # 假設處理單頁
        mapped_entities = []

        for entity in entities:
            try:
                # 尋找包含該實體的文字區塊
                containing_block = self._find_containing_block(
                    entity, page_layout.blocks
                )

                if containing_block and hasattr(containing_block, 'bbox'):
                    # 計算實體在區塊內的相對位置
                    relative_bbox = self._calculate_entity_bbox(
                        entity, containing_block
                    )

                    if relative_bbox:
                        # 映射到圖像座標
                        image_bbox = self._map_to_image_coords(
                            relative_bbox, image_shape
                        )

                        # 創建新的實體對象，包含圖像座標
                        mapped_entity = Entity(
                            text=entity.text,
                            entity_type=entity.entity_type,
                            start=entity.start,
                            end=entity.end,
                            confidence=entity.confidence
                        )

                        # 添加圖像座標
                        mapped_entity.image_bbox = image_bbox
                        mapped_entity.text_block = containing_block

                        mapped_entities.append(mapped_entity)

                        if self.debug_mode:
                            logger.debug(f"實體 '{entity.text}' 映射到座標: {image_bbox}")
                    else:
                        # 無法精確定位，保留原實體
                        mapped_entities.append(entity)
                else:
                    # 無法找到包含區塊，保留原實體
                    mapped_entities.append(entity)

            except Exception as e:
                logger.warning(f"實體座標映射失敗: {e}")
                mapped_entities.append(entity)

        return mapped_entities

    def _find_containing_block(
        self,
        entity: Entity,
        blocks: List[TextBlock]
    ) -> Optional[TextBlock]:
        """尋找包含指定實體的文字區塊"""
        for block in blocks:
            if hasattr(block, 'text') and hasattr(block, 'char_start'):
                # 檢查實體是否在此區塊的文字範圍內
                if (block.char_start <= entity.start <
                    block.char_start + len(block.text)):
                    return block

        return None

    def _calculate_entity_bbox(
        self,
        entity: Entity,
        text_block: TextBlock
    ) -> Optional[Tuple[int, int, int, int]]:
        """計算實體在文字區塊內的邊界框"""
        if not hasattr(text_block, 'bbox') or not hasattr(text_block, 'text'):
            return None

        try:
            # 簡化計算：基於字符位置估算
            block_text = text_block.text
            block_bbox = text_block.bbox  # (x1, y1, x2, y2)

            # 計算實體在區塊文字中的相對位置
            entity_start_in_block = entity.start - getattr(text_block, 'char_start', 0)
            entity_end_in_block = entity.end - getattr(text_block, 'char_start', 0)

            # 確保範圍有效
            entity_start_in_block = max(0, entity_start_in_block)
            entity_end_in_block = min(len(block_text), entity_end_in_block)

            if entity_start_in_block >= entity_end_in_block:
                return None

            # 基於文字比例估算座標
            text_ratio_start = entity_start_in_block / len(block_text)
            text_ratio_end = entity_end_in_block / len(block_text)

            # 計算實體邊界框 (簡化為水平排列)
            x1, y1, x2, y2 = block_bbox
            entity_x1 = int(x1 + (x2 - x1) * text_ratio_start)
            entity_x2 = int(x1 + (x2 - x1) * text_ratio_end)

            return (entity_x1, y1, entity_x2, y2)

        except Exception as e:
            logger.debug(f"實體座標計算失敗: {e}")
            return None

    def _map_to_image_coords(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """將相對座標映射到圖像座標"""
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox

        # 確保座標在圖像範圍內
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        return (x1, y1, x2, y2)

    def _apply_deidentification(
        self,
        image: np.ndarray,
        text: str,
        entities: List[Entity]
    ) -> Tuple[np.ndarray, str]:
        """執行去識別化處理"""
        processed_image = image.copy()
        processed_text = text

        # 文字去識別化
        if entities:
            processed_text = self.replacer.replace_entities(text, entities)

        # 圖像去識別化
        for entity in entities:
            if hasattr(entity, 'image_bbox'):
                x1, y1, x2, y2 = entity.image_bbox

                if self.replacement_mode == "mask":
                    # 黑條遮蔽
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
                elif self.replacement_mode == "fake":
                    # 假資料替換 (在圖像上繪製假文字)
                    fake_text = self.fake_provider.generate_fake_value(
                        entity.entity_type, entity.text
                    )
                    self._draw_fake_text(processed_image, fake_text, (x1, y1, x2, y2))

        return processed_image, processed_text

    def _draw_fake_text(
        self,
        image: np.ndarray,
        fake_text: str,
        bbox: Tuple[int, int, int, int]
    ):
        """在圖像上繪製假文字"""
        x1, y1, x2, y2 = bbox

        # 先遮蔽原區域
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # 計算文字大小和位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(1.0, (x2 - x1) / (len(fake_text) * 10))
        thickness = max(1, int(font_scale * 2))

        # 獲取文字大小
        (text_width, text_height), baseline = cv2.getTextSize(
            fake_text, font, font_scale, thickness
        )

        # 計算文字位置 (置中)
        text_x = x1 + max(0, (x2 - x1 - text_width) // 2)
        text_y = y1 + max(text_height, (y2 - y1 + text_height) // 2)

        # 繪製文字
        cv2.putText(
            image, fake_text, (text_x, text_y),
            font, font_scale, (0, 0, 0), thickness
        )

    def _create_empty_result(self, image: np.ndarray, image_name: str) -> Dict:
        """創建空結果"""
        return {
            "success": True,
            "image_name": image_name,
            "original_text": "",
            "processed_text": "",
            "entities": [],
            "processing_info": {
                "entities_found": 0,
                "replacement_mode": self.replacement_mode
            },
            "images": {
                "processed": image  # 原圖不變
            }
        }

    def _create_result(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray,
        original_text: str,
        processed_text: str,
        entities: List[Entity],
        image_name: str
    ) -> Dict:
        """創建處理結果"""
        return {
            "success": True,
            "image_name": image_name,
            "original_text": original_text,
            "processed_text": processed_text,
            "entities": [self._entity_to_dict(e) for e in entities],
            "processing_info": {
                "entities_found": len(entities),
                "replacement_mode": self.replacement_mode,
                "detector_status": self.pii_detector.get_detector_status()
            },
            "images": {
                "original": original_image,
                "processed": processed_image,
                "annotated": self._create_annotated_image(original_image, entities)
            }
        }

    def _create_annotated_image(
        self,
        image: np.ndarray,
        entities: List[Entity]
    ) -> np.ndarray:
        """創建標註圖像"""
        annotated = image.copy()

        # 定義實體類型顏色
        colors = {
            "PERSON": (0, 255, 0),
            "ID_NUMBER": (255, 0, 0),
            "PHONE_NUMBER": (0, 0, 255),
            "EMAIL_ADDRESS": (255, 255, 0),
            "ADDRESS": (255, 0, 255),
            "default": (128, 128, 128)
        }

        for entity in entities:
            if hasattr(entity, 'image_bbox'):
                x1, y1, x2, y2 = entity.image_bbox
                color = colors.get(entity.entity_type, colors["default"])

                # 繪製邊界框
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # 繪製標籤
                label = f"{entity.entity_type}({entity.confidence:.2f})"
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        return annotated

    def _entity_to_dict(self, entity: Entity) -> Dict:
        """將實體轉換為字典格式"""
        result = {
            "text": entity.text,
            "entity_type": entity.entity_type,
            "start": entity.start,
            "end": entity.end,
            "confidence": entity.confidence
        }

        if hasattr(entity, 'image_bbox'):
            result["image_bbox"] = entity.image_bbox

        if hasattr(entity, 'sources'):
            result["sources"] = entity.sources

        return result

    def _save_results(self, result: Dict, output_path: str):
        """保存處理結果"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_name = result["image_name"]
        base_name = Path(image_name).stem

        # 保存處理後的圖像
        if "processed" in result["images"]:
            processed_path = output_dir / f"{base_name}_processed.jpg"
            cv2.imwrite(str(processed_path), result["images"]["processed"])

        # 保存標註圖像
        if "annotated" in result["images"]:
            annotated_path = output_dir / f"{base_name}_annotated.jpg"
            cv2.imwrite(str(annotated_path), result["images"]["annotated"])

        # 保存處理資訊
        import json
        info_path = output_dir / f"{base_name}_info.json"

        info_data = {
            "image_name": result["image_name"],
            "entities": result["entities"],
            "processing_info": result["processing_info"],
            "original_text": result["original_text"],
            "processed_text": result["processed_text"]
        }

        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, ensure_ascii=False, indent=2)

        logger.info(f"結果已保存到: {output_dir}")

    def _log_component_status(self):
        """記錄組件狀態 (除錯模式)"""
        logger.debug("=== 組件狀態 ===")
        logger.debug(f"文字抽取器: {type(self.text_extractor).__name__}")
        logger.debug(f"PII 偵測器: {self.pii_detector.get_detector_status()}")
        logger.debug(f"替換模式: {self.replacement_mode}")
        logger.debug("==================")

    def batch_process(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: str
    ) -> List[Dict]:
        """批次處理多張圖像"""
        results = []

        for i, image_path in enumerate(input_paths):
            logger.info(f"處理第 {i+1}/{len(input_paths)} 張圖像: {image_path}")

            try:
                result = self.process_image(image_path, output_dir)
                results.append(result)

            except Exception as e:
                logger.error(f"圖像 {image_path} 處理失敗: {e}")
                results.append({
                    "success": False,
                    "image_name": str(image_path),
                    "error": str(e)
                })

        return results
