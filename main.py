# main.py
"""
EdgeDeID Studio 主程式
整合 BERT NER 偵測器的完整使用範例
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json
from src.deid_pipeline.pii.utils.base import Entity, PIIDetector
from src.deid_pipeline.config import Config, validate_config, get_detector_config
from src.deid_pipeline.image_deid.processor import ImageDeidProcessor
from src.deid_pipeline.pii.detectors.composite import CompositeDetector, DetectorConfig
from scripts.download_models import ModelDownloader


# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 導入專案模組
try:
    logger.info("模組導入成功")

except ImportError as e:
    logger.error(f"模組導入失敗: {e}")
    logger.error("請確保已正確安裝所有相依套件並設定 PYTHONPATH")
    sys.exit(1)

class EdgeDeIDStudio:
    """EdgeDeID Studio 主要控制類別"""

    def __init__(self, use_onnx: bool = True, debug_mode: bool = False):
        """
        初始化 EdgeDeID Studio

        Args:
            use_onnx: 是否使用 ONNX 模型 (推薦)
            debug_mode: 除錯模式
        """
        self.use_onnx = use_onnx
        self.debug_mode = debug_mode

        # 驗證配置
        try:
            validate_config()
            logger.info("配置驗證通過")
        except ValueError as e:
            logger.error(f"配置錯誤: {e}")
            if not Config.USE_STUB:
                raise

        # 初始化組件
        self._initialize_components()

    def _initialize_components(self):
        """初始化各個組件"""
        logger.info("初始化 EdgeDeID Studio 組件...")

        # 1. 初始化圖像處理器
        self.image_processor = ImageDeidProcessor(
            use_onnx=self.use_onnx,
            ocr_engine="auto",
            replacement_mode="mask",
            debug_mode=self.debug_mode
        )

        # 2. 初始化獨立的文字偵測器 (供非圖像處理使用)
        detector_configs = self._create_detector_configs()
        self.text_detector = CompositeDetector(
            detector_configs=detector_configs,
            merge_strategy="weighted_vote",
            confidence_threshold=0.6
        )

        logger.info("組件初始化完成")

    def _create_detector_configs(self) -> List[DetectorConfig]:
        """創建偵測器配置"""
        configs = []

        # Regex 偵測器
        configs.append(DetectorConfig(
            detector_type="regex",
            enabled=True,
            weight=0.8,
            config={"use_zh_rules": True, "use_en_rules": True}
        ))

        # BERT 偵測器
        bert_config = get_detector_config("bert_onnx" if self.use_onnx else "bert_pytorch")
        configs.append(DetectorConfig(
            detector_type="bert_onnx" if self.use_onnx else "bert_pytorch",
            enabled=not Config.USE_STUB,
            weight=1.3 if self.use_onnx else 1.2,
            config=bert_config
        ))

        return configs

    def setup_models(self, force_download: bool = False):
        """設定和下載模型"""
        if Config.USE_STUB:
            logger.info("STUB 模式，跳過模型設定")
            return

        logger.info("開始模型設定...")

        try:
            downloader = ModelDownloader()

            # 下載中文 BERT NER 模型
            logger.info("下載中文 BERT NER 模型...")
            downloader.download_model("bert-base-chinese-ner", force_download)

            # 轉換為 ONNX (如果使用 ONNX)
            if self.use_onnx:
                logger.info("轉換為 ONNX 格式...")
                onnx_path = downloader.convert_to_onnx("bert-base-chinese-ner")
                logger.info(f"ONNX 模型路徑: {onnx_path}")

            logger.info("模型設定完成")

        except Exception as e:
            logger.error(f"模型設定失敗: {e}")
            if not Config.USE_STUB:
                raise

    def process_image(self, image_path: str, output_dir: Optional[str] = None) -> dict:
        """
        處理單張圖像

        Args:
            image_path: 圖像檔案路徑
            output_dir: 輸出目錄

        Returns:
            處理結果
        """
        logger.info(f"處理圖像: {image_path}")

        try:
            result = self.image_processor.process_image(image_path, output_dir)

            if result.get("success"):
                logger.info(f"圖像處理成功，發現 {len(result.get('entities', []))} 個 PII 實體")
            else:
                logger.error(f"圖像處理失敗: {result.get('error', '未知錯誤')}")

            return result

        except Exception as e:
            logger.error(f"圖像處理異常: {e}")
            return {"success": False, "error": str(e)}

    def process_text(self, text: str) -> dict:
        """
        處理純文字

        Args:
            text: 輸入文字

        Returns:
            處理結果
        """
        logger.info("處理文字輸入")

        try:
            entities = self.text_detector.detect(text)

            result = {
                "success": True,
                "original_text": text,
                "entities": [self._entity_to_dict(e) for e in entities],
                "entities_count": len(entities)
            }

            logger.info(f"文字處理完成，發現 {len(entities)} 個 PII 實體")
            return result

        except Exception as e:
            logger.error(f"文字處理失敗: {e}")
            return {"success": False, "error": str(e)}

    def batch_process_images(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.jpg"
    ) -> List[dict]:
        """
        批次處理圖像

        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄
            file_pattern: 檔案模式

        Returns:
            處理結果列表
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"輸入目錄不存在: {input_dir}")

        # 找到所有符合條件的檔案
        image_files = list(input_path.glob(file_pattern))
        logger.info(f"找到 {len(image_files)} 個圖像檔案")

        if not image_files:
            logger.warning("未找到符合條件的圖像檔案")
            return []

        # 批次處理
        results = self.image_processor.batch_process(image_files, output_dir)

        # 統計結果
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"批次處理完成: {success_count}/{len(results)} 成功")

        return results

    def benchmark(self, test_texts: Optional[List[str]] = None) -> dict:
        """
        效能基準測試

        Args:
            test_texts: 測試文字列表

        Returns:
            測試結果
        """
        if test_texts is None:
            test_texts = Config.BENCHMARK_TEXTS

        logger.info("開始效能基準測試...")

        try:
            # 測試偵測器效能
            detector_results = self.text_detector.benchmark_detectors(test_texts)

            # 測試圖像處理器效能 (如果有測試圖像)
            image_results = {}
            test_image_path = Path("data/test/sample_doc.png")
            if test_image_path.exists():
                import time
                start_time = time.time()
                self.process_image(str(test_image_path))
                end_time = time.time()

                image_results = {
                    "image_processing_time": end_time - start_time
                }

            result = {
                "detector_performance": detector_results,
                "image_processing": image_results,
                "test_config": {
                    "use_onnx": self.use_onnx,
                    "use_stub": Config.USE_STUB,
                    "test_texts_count": len(test_texts)
                }
            }

            logger.info("效能測試完成")
            return result

        except Exception as e:
            logger.error(f"效能測試失敗: {e}")
            return {"error": str(e)}

    def get_system_status(self) -> dict:
        """獲取系統狀態"""
        try:
            status = {
                "system_info": {
                    "use_onnx": self.use_onnx,
                    "use_stub": Config.USE_STUB,
                    "debug_mode": self.debug_mode
                },
                "detector_status": self.text_detector.get_detector_status(),
                "supported_entities": self.text_detector.get_supported_entities()
            }

            # 獲取模型資訊
            for detector_name, detector in self.text_detector.detectors.items():
                if hasattr(detector, 'get_model_info'):
                    status[f"{detector_name}_model_info"] = detector.get_model_info()

            return status

        except Exception as e:
            logger.error(f"獲取系統狀態失敗: {e}")
            return {"error": str(e)}

    def _entity_to_dict(self, entity: Entity) -> dict:
        """將實體轉換為字典格式"""
        return {
            "text": entity.text,
            "entity_type": entity.entity_type,
            "start": entity.start,
            "end": entity.end,
            "confidence": entity.confidence,
            "sources": getattr(entity, 'sources', [])
        }

def main():
    """主要執行函數"""
    parser = argparse.ArgumentParser(description="EdgeDeID Studio - PII 去識別化工具")

    # 基本選項
    parser.add_argument("--setup", action="store_true", help="設定和下載模型")
    parser.add_argument("--use-pytorch", action="store_true", help="使用 PyTorch 而非 ONNX")
    parser.add_argument("--debug", action="store_true", help="啟用除錯模式")

    # 處理選項
    parser.add_argument("--image", type=str, help="處理單張圖像")
    parser.add_argument("--text", type=str, help="處理文字輸入")
    parser.add_argument("--batch", type=str, help="批次處理圖像目錄")
    parser.add_argument("--output", type=str, help="輸出目錄", default="output")

    # 測試選項
    parser.add_argument("--benchmark", action="store_true", help="執行效能測試")
    parser.add_argument("--status", action="store_true", help="顯示系統狀態")

    # 其他選項
    parser.add_argument("--force-download", action="store_true", help="強制重新下載模型")

    args = parser.parse_args()

    try:
        # 初始化系統
        use_onnx = not args.use_pytorch
        studio = EdgeDeIDStudio(use_onnx=use_onnx, debug_mode=args.debug)

        # 模型設定
        if args.setup:
            studio.setup_models(force_download=args.force_download)
            return

        # 系統狀態
        if args.status:
            status = studio.get_system_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return

        # 效能測試
        if args.benchmark:
            results = studio.benchmark()
            print(json.dumps(results, indent=2, ensure_ascii=False))
            return

        # 圖像處理
        if args.image:
            result = studio.process_image(args.image, args.output)
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return

        # 文字處理
        if args.text:
            result = studio.process_text(args.text)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        # 批次處理
        if args.batch:
            results = studio.batch_process_images(args.batch, args.output)
            print(f"批次處理完成，共處理 {len(results)} 個檔案")

            # 保存結果摘要
            output_path = Path(args.output)
            summary_path = output_path / "batch_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            print(f"結果摘要已保存至: {summary_path}")
            return

        # 如果沒有指定動作，顯示幫助
        parser.print_help()

    except KeyboardInterrupt:
        logger.info("使用者中斷操作")
    except Exception as e:
        logger.error(f"執行失敗: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
