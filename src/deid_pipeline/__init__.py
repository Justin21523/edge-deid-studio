# src/deid_pipeline/__init__.py
from .parser.text_extractor import extract_text
from .image_deid.processor import ImageDeidProcessor
from .pii.detectors import get_detector
from .pii.utils.replacer import Replacer

class DeidResult:
    """
    把偵測結果包成 result.entities 及 result.text
    供原始測試呼叫。
    """
    def __init__(self, entities, text):
        self.entities = entities
        self.text = text

class DeidPipeline:
    """
    統一介面：.process(input_path, output_mode, generate_report)
    不同副檔用 extract_text 或 OCRPIIProcessor。
    """
    def __init__(self, language: str = "zh"):
        self.lang = language
        self.detector = get_detector(language)
        self.replacer = Replacer()
        self.ocr_proc = ImageDeidProcessor(lang=language)

    def process(self, input_path: str, output_mode: str = "replacement", generate_report: bool = False):
        # 1. 讀文字或影像
        suffix = input_path.lower().split(".")[-1]
        if suffix in ("txt", "docx", "pdf"):
            text, _ = extract_text(input_path)
        else:
            ocr_res = self.ocr_proc.process_image(input_path)
            text = ocr_res.get("original_text", "")

        # 2. 偵測
        entities = self.detector.detect(text)

        # 3. 替換
        clean_text, _ = self.replacer.replace(text, entities)

        # 4. 回傳
        return DeidResult(entities=entities, text=clean_text)


#!/usr/bin/env python3
"""
EdgeDeID Studio 完整串接實作示例
展示 replacer.py 和 fake_provider.py 與其他模組的串接
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time

# 添加項目根目錄到 Python 路徑
sys.path.append(str(Path(__file__).parent))

from src.deid_pipeline.config import Config
from src.deid_pipeline.base import Entity
from src.deid_pipeline.parser.text_extractor import SmartTextExtractor
from src.deid_pipeline.parser.ocr import OCRAdapter
from src.deid_pipeline.parser.position_mapper import TextPositionMapper
from src.deid_pipeline.pii.composite import CompositeDetector
from src.deid_pipeline.pii.regex_detector import RegexDetector
from src.deid_pipeline.pii.bert_onnx_detector import BertONNXNERDetector
from src.deid_pipeline.utils.replacer import Replacer, ReplacementMode
from src.deid_pipeline.utils.fake_provider import FakeProvider
from src.deid_pipeline.image_deid.processor import ImageDeidProcessor

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EdgeDeidSystem:
    """
    EdgeDeID 完整系統整合類
    展示所有模組的正確串接方式
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化系統"""
        self.config = config or Config()
        self.is_initialized = False

        # 核心組件
        self.text_extractor: Optional[SmartTextExtractor] = None
        self.pii_detector: Optional[CompositeDetector] = None
        self.replacer: Optional[Replacer] = None
        self.fake_provider: Optional[FakeProvider] = None
        self.processor: Optional[ImageDeidProcessor] = None

        # 性能統計
        self.stats = {
            'files_processed': 0,
            'total_entities_found': 0,
            'total_entities_replaced': 0,
            'total_processing_time': 0.0
        }

    def initialize(self) -> bool:
        """初始化所有組件"""
        try:
            logger.info("開始初始化 EdgeDeID 系統...")

            # 1. 檢查模型文件
            if not self._check_models():
                logger.error("模型文件檢查失敗，請先下載模型")
                return False

            # 2. 初始化 OCR 和文字抽取器
            self._initialize_text_extraction()

            # 3. 初始化 PII 偵測器
            self._initialize_pii_detection()

            # 4. 初始化假資料生成器
            self._initialize_fake_generation()

            # 5. 初始化替換器
            self._initialize_replacer()

            # 6. 初始化主處理器
            self._initialize_processor()

            self.is_initialized = True
            logger.info("✓ EdgeDeID 系統初始化完成")
            return True

        except Exception as e:
            logger.error(f"系統初始化失敗: {e}")
            return False

    def _check_models(self) -> bool:
        """檢查必要的模型文件"""
        required_models = [
            self.config.GPT2_MODEL_PATH,
            self.config.BERT_NER_MODEL_PATH
        ]

        for model_path in required_models:
            if not Path(model_path).exists():
                logger.error(f"模型路徑不存在: {model_path}")
                return False

        return True

    def _initialize_text_extraction(self):
        """初始化文字抽取組件"""
        logger.info("初始化文字抽取組件...")

        # OCR 適配器
        ocr_adapter = OCRAdapter(
            use_layout_model=self.config.USE_LAYOUT_MODEL,
            layout_model_path=self.config.LAYOUT_MODEL_PATH,
            deskew_enabled=True,
            clahe_enabled=True,
            min_confidence=self.config.OCR_MIN_CONFIDENCE
        )

        # 文字抽取器
        self.text_extractor = SmartTextExtractor(
            ocr_adapter=ocr_adapter,
            ocr_threshold=self.config.OCR_THRESHOLD
        )

        logger.info("✓ 文字抽取組件初始化完成")

    def _initialize_pii_detection(self):
        """初始化 PII 偵測組件"""
        logger.info("初始化 PII 偵測組件...")

        detectors = []

        # 正則偵測器
        if self.config.USE_REGEX_DETECTOR:
            regex_detector = RegexDetector(
                chinese_rules_path=self.config.REGEX_ZH_PATH,
                english_rules_path=self.config.REGEX_EN_PATH
            )
            detectors.append(regex_detector)
            logger.info("✓ 已載入正則偵測器")

        # BERT ONNX 偵測器
        if self.config.USE_BERT_DETECTOR:
            bert_detector = BertONNXNERDetector(
                model_path=self.config.BERT_NER_MODEL_PATH,
                confidence_threshold=self.config.BERT_CONFIDENCE_THRESHOLD
            )
            detectors.append(bert_detector)
            logger.info("✓ 已載入 BERT ONNX 偵測器")

        # 組合偵測器
        self.pii_detector = CompositeDetector(
            detectors=detectors,
            entity_priority=self.config.ENTITY_PRIORITY
        )

        logger.info("✓ PII 偵測組件初始化完成")

    def _initialize_fake_generation(self):
        """初始化假資料生成組件"""
        logger.info("初始化假資料生成組件...")

        self.fake_provider = FakeProvider(
            gpt2_model_path=self.config.GPT2_MODEL_PATH,
            chinese_model_path=self.config.GPT2_CHINESE_MODEL_PATH,
            use_gpu=self.config.USE_GPU
        )

        logger.info("✓ 假資料生成組件初始化完成")

    def _initialize_replacer(self):
        """初始化替換組件"""
        logger.info("初始化替換組件...")

        replacement_mode = ReplacementMode(self.config.DEFAULT_REPLACEMENT_MODE)

        self.replacer = Replacer(
            mode=replacement_mode,
            fake_provider=self.fake_provider,
            preserve_format=self.config.PRESERVE_FORMAT,
            consistency_cache=self.config.USE_CONSISTENCY_CACHE
        )

        logger.info("✓ 替換組件初始化完成")

    def _initialize_processor(self):
        """初始化主處理器"""
        logger.info("初始化主處理器...")

        self.processor = ImageDeidProcessor(
            text_extractor=self.text_extractor,
            pii_detector=self.pii_detector,
            replacer=self.replacer,
            debug_mode=self.config.DEBUG_MODE
        )

        logger.info("✓ 主處理器初始化完成")

    def process_file(self,
                    input_path: str,
                    output_path: Optional[str] = None,
                    replacement_mode: Optional[str] = None) -> Dict:
        """
        處理單個文件

        Args:
            input_path: 輸入文件路徑
            output_path: 輸出文件路徑
            replacement_mode: 替換模式 (mask/redact/fake/partial)

        Returns:
            處理結果字典
        """
        if not self.is_initialized:
            raise RuntimeError("系統尚未初始化，請先調用 initialize()")

        start_time = time.time()

        try:
            logger.info(f"開始處理文件: {input_path}")

            # 設置替換模式
            if replacement_mode:
                mode = ReplacementMode(replacement_mode)
            else:
                mode = None

            # 處理文件
            if Path(input_path).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # 圖像文件
                result = self.processor.process_image(input_path)
            else:
                # 文字文件
                result = self._process_text_file(input_path, mode)

            # 保存結果
            if output_path:
                self._save_result(result, output_path, input_path)

            # 更新統計
            processing_time = time.time() - start_time
            self._update_stats(result, processing_time)

            logger.info(f"✓ 文件處理完成: {input_path} ({processing_time:.2f}s)")

            return {
                'status': 'success',
                'input_path': input_path,
                'output_path': output_path,
                'entities_found': len(result.entities_found),
                'entities_replaced': len(result.entities_replaced),
                'processing_time': processing_time,
                'original_text_length': len(result.original_text),
                'processed_text_length': len(result.processed_text)
            }

        except Exception as e:
            logger.error(f"處理文件失敗 {input_path}: {e}")
            return {
                'status': 'error',
                'input_path': input_path,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def _process_text_file(self, file_path: str, mode: Optional[ReplacementMode]):
        """處理純文字文件"""
        # 1. 抽取文字
        document_layout = self.text_extractor.extract_text(file_path)
        full_text = document_layout.get_text()

        # 2. PII 偵測
        entities = self.pii_detector.detect(full_text)

        # 3. 替換處理
        replacement_result = self.replacer.replace_entities(
            full_text, entities, mode
        )

        # 4. 創建結果對象
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class ProcessingResult:
            original_text: str
            processed_text: str
            entities_found: List[Entity]
            entities_replaced: List[Entity]
            replacement_map: Dict[str, str]
            metadata: Dict[str, Any]

        return ProcessingResult(
            original_text=full_text,
            processed_text=replacement_result.processed_text,
            entities_found=entities,
            entities_replaced=replacement_result.entities_replaced,
            replacement_map=replacement_result.replacement_map,
            metadata={'file_type': 'text', 'document_layout': document_layout}
        )

    def _save_result(self, result, output_path: str, input_path: str):
        """保存處理結果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 根據輸入文件類型決定輸出格式
        input_ext = Path(input_path).suffix.lower()

        if input_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # 圖像文件 - 保存標註版本和文字
            if hasattr(result, 'save_annotated_image'):
                result.save_annotated_image(str(output_path))

            # 同時保存文字版本
            text_output = output_path.with_suffix('.txt')
            with open(text_output, 'w', encoding='utf-8') as f:
                f.write(result.processed_text)

        else:
            # 文字文件 - 保存處理後的文字
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.processed_text)

        # 保存處理報告
        report_path = output_path.with_suffix('.report.json')
        self._save_processing_report(result, report_path)

    def _save_processing_report(self, result, report_path: Path):
        """保存處理報告"""
        import json

        report = {
            'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_text_length': len(result.original_text),
            'processed_text_length': len(result.processed_text),
            'entities_found': [
                {
                    'text': entity.text,
                    'type': entity.entity_type,
                    'start': entity.start,
                    'end': entity.end,
                    'confidence': entity.confidence,
                    'detector': entity.detector_name
                }
                for entity in result.entities_found
            ],
            'entities_replaced': len(result.entities_replaced),
            'replacement_summary': {
                entity_type: len([e for e in result.entities_replaced if e.entity_type == entity_type])
                for entity_type in set(e.entity_type for e in result.entities_replaced)
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def _update_stats(self, result, processing_time: float):
        """更新統計信息"""
        self.stats['files_processed'] += 1
        self.stats['total_entities_found'] += len(result.entities_found)
        self.stats['total_entities_replaced'] += len(result.entities_replaced)
        self.stats['total_processing_time'] += processing_time

    def batch_process(self,
                     input_dir: str,
                     output_dir: str,
                     file_patterns: List[str] = None) -> List[Dict]:
        """
        批次處理文件

        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄
            file_patterns: 文件模式列表，如 ['*.pdf', '*.docx', '*.jpg']

        Returns:
            處理結果列表
        """
        if not self.is_initialized:
            raise RuntimeError("系統尚未初始化，請先調用 initialize()")

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 預設文件模式
        if file_patterns is None:
            file_patterns = ['*.pdf', '*.docx', '*.txt', '*.jpg', '*.png', '*.csv', '*.xlsx']

        # 收集文件
        files_to_process = []
        for pattern in file_patterns:
            files_to_process.extend(input_path.glob(pattern))

        logger.info(f"找到 {len(files_to_process)} 個文件需要處理")

        results = []
        for file_path in files_to_process:
            # 生成輸出路徑
            relative_path = file_path.relative_to(input_path)
            output_file = output_path / f"{relative_path.stem}_deid{relative_path.suffix}"

            # 處理文件
            result = self.process_file(str(file_path), str(output_file))
            results.append(result)

        # 輸出批次處理統計
        self._print_batch_stats(results)

        return results

    def _print_batch_stats(self, results: List[Dict]):
        """輸出批次處理統計"""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']

        total_entities = sum(r.get('entities_found', 0) for r in successful)
        total_replaced = sum(r.get('entities_replaced', 0) for r in successful)
        total_time = sum(r.get('processing_time', 0) for r in results)

        print("\n" + "="*50)
        print("批次處理統計")
        print("="*50)
        print(f"總文件數: {len(results)}")
        print(f"成功處理: {len(successful)}")
        print(f"處理失敗: {len(failed)}")
        print(f"偵測到的PII實體總數: {total_entities}")
        print(f"替換的PII實體總數: {total_replaced}")
        print(f"總處理時間: {total_time:.2f}s")
        print(f"平均處理時間: {total_time/len(results):.2f}s")
        print("="*50)

    def demo_different_modes(self, test_text: str):
        """演示不同替換模式的效果"""
        if not self.is_initialized:
            raise RuntimeError("系統尚未初始化，請先調用 initialize()")

        print("\n" + "="*60)
        print("不同替換模式演示")
        print("="*60)
        print(f"原始文字: {test_text}")
        print("-"*60)

        # 偵測 PII 實體
        entities = self.pii_detector.detect(test_text)
        print(f"偵測到 {len(entities)} 個PII實體:")
        for entity in entities:
            print(f"  - {entity.entity_type}: '{entity.text}' (信心度: {entity.confidence:.2f})")
        print("-"*60)

        # 測試不同模式
        modes = [
            (ReplacementMode.MASK, "遮蔽模式"),
            (ReplacementMode.REDACT, "隱碼模式"),
            (ReplacementMode.PARTIAL, "部分遮蔽模式"),
            (ReplacementMode.FAKE, "假資料替換模式")
        ]

        for mode, mode_name in modes:
            result = self.replacer.replace_entities(test_text, entities, mode)
            print(f"{mode_name}: {result.processed_text}")

        print("="*60)

    def get_system_stats(self) -> Dict:
        """獲取系統統計信息"""
        stats = self.stats.copy()

        if self.replacer:
            replacer_stats = self.replacer.get_statistics()
            stats.update({f"replacer_{k}": v for k, v in replacer_stats.items()})

        if self.fake_provider:
            cache_stats = self.fake_provider.get_cache_stats()
            stats.update({f"fake_provider_{k}": v for k, v in cache_stats.items()})

        return stats

    def cleanup(self):
        """清理系統資源"""
        if self.replacer:
            self.replacer.clear_cache()

        if self.fake_provider:
            self.fake_provider.clear_cache()

        logger.info("系統資源清理完成")


def main():
    """主函數 - 展示完整使用流程"""

    # 1. 初始化系統
    print("正在初始化 EdgeDeID 系統...")
    system = EdgeDeidSystem()

    if not system.initialize():
        print("❌ 系統初始化失敗")
        return

    print("✅ 系統初始化成功")

    # 2. 演示不同替換模式
    test_text = "我是張小明，電話號碼是0912345678，電子郵件是zhang.xiaoming@example.com，住址是台北市大安區忠孝東路123號"
    system.demo_different_modes(test_text)

    # 3. 處理單個文件示例
    print("\n處理單個文件示例:")

    # 創建測試文件
    test_file = "test_input/sample.txt"
    os.makedirs("test_input", exist_ok=True)

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_text)

    # 處理文件
    result = system.process_file(
        input_path=test_file,
        output_path="test_output/sample_deid.txt",
        replacement_mode="fake"
    )

    print(f"處理結果: {result}")

    # 4. 批次處理示例 (如果有測試文件的話)
    test_dir = Path("test_input")
    if test_dir.exists() and any(test_dir.iterdir()):
        print("\n批次處理示例:")
        batch_results = system.batch_process(
            input_dir="test_input",
            output_dir="test_output"
        )

    # 5. 顯示系統統計
    print("\n系統統計信息:")
    stats = system.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 6. 清理資源
    system.cleanup()
    print("\n✅ 演示完成")


class ConfigurationManager:
    """配置管理器 - 展示如何動態調整系統參數"""

    @staticmethod
    def create_optimized_config(use_case: str) -> Config:
        """根據使用情境創建優化配置"""
        config = Config()

        if use_case == "speed":
            # 速度優先配置
            config.USE_LAYOUT_MODEL = False
            config.USE_BERT_DETECTOR = False
            config.DEFAULT_REPLACEMENT_MODE = "MASK"
            config.OCR_MIN_CONFIDENCE = 0.6

        elif use_case == "accuracy":
            # 準確度優先配置
            config.USE_LAYOUT_MODEL = True
            config.USE_BERT_DETECTOR = True
            config.BERT_CONFIDENCE_THRESHOLD = 0.9
            config.DEFAULT_REPLACEMENT_MODE = "FAKE"
            config.OCR_MIN_CONFIDENCE = 0.8

        elif use_case == "privacy":
            # 隱私優先配置
            config.DEFAULT_REPLACEMENT_MODE = "FAKE"
            config.USE_CONSISTENCY_CACHE = True
            config.PRESERVE_FORMAT = True

        return config

    @staticmethod
    def validate_config(config: Config) -> List[str]:
        """驗證配置的有效性"""
        issues = []

        # 檢查模型路徑
        if config.USE_BERT_DETECTOR and not Path(config.BERT_NER_MODEL_PATH).exists():
            issues.append(f"BERT模型路徑不存在: {config.BERT_NER_MODEL_PATH}")

        # 檢查閾值設置
        if not 0.0 <= config.BERT_CONFIDENCE_THRESHOLD <= 1.0:
            issues.append("BERT信心閾值必須在0.0-1.0之間")

        if not 0.0 <= config.OCR_MIN_CONFIDENCE <= 1.0:
            issues.append("OCR最小信心度必須在0.0-1.0之間")

        return issues


class PerformanceProfiler:
    """性能分析器 - 用於系統性能監控和優化"""

    def __init__(self):
        self.profiles = {}

    def profile_component(self, component_name: str, func, *args, **kwargs):
        """分析組件性能"""
        import time
        import psutil

        # 記錄開始狀態
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # 執行函數
        result = func(*args, **kwargs)

        # 記錄結束狀態
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        # 存儲性能數據
        self.profiles[component_name] = {
            'execution_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return result

    def get_performance_report(self) -> str:
        """獲取性能報告"""
        report = ["性能分析報告", "="*40]

        for component, data in self.profiles.items():
            report.append(f"{component}:")
            report.append(f"  執行時間: {data['execution_time']:.4f}s")
            report.append(f"  記憶體佔用: {data['memory_usage']/1024/1024:.2f}MB")
            report.append(f"  記錄時間: {data['timestamp']}")
            report.append("")

        return "\n".join(report)


def advanced_usage_example():
    """進階使用示例"""

    print("進階使用示例")
    print("="*50)

    # 1. 使用不同配置
    configs = {
        "速度優先": ConfigurationManager.create_optimized_config("speed"),
        "準確度優先": ConfigurationManager.create_optimized_config("accuracy"),
        "隱私優先": ConfigurationManager.create_optimized_config("privacy")
    }

    for config_name, config in configs.items():
        print(f"\n{config_name}配置:")

        # 驗證配置
        issues = ConfigurationManager.validate_config(config)
        if issues:
            print("  配置問題:")
            for issue in issues:
                print(f"    - {issue}")
            continue

        # 初始化系統
        system = EdgeDeidSystem(config)
        if not system.initialize():
            print("  ❌ 初始化失敗")
            continue

        # 性能測試
        profiler = PerformanceProfiler()

        test_text = "測試文字：李大華，電話0987654321，email: li@test.com"

        # 分析各組件性能
        entities = profiler.profile_component(
            "PII偵測",
            system.pii_detector.detect,
            test_text
        )

        replacement_result = profiler.profile_component(
            "文字替換",
            system.replacer.replace_entities,
            test_text, entities
        )

        print(f"  ✅ 初始化成功")
        print(f"  偵測到 {len(entities)} 個實體")
        print(f"  替換了 {len(replacement_result.entities_replaced)} 個實體")

        # 輸出性能報告
        print(profiler.get_performance_report())

        system.cleanup()


def troubleshooting_guide():
    """故障排除指南"""

    print("EdgeDeID 故障排除指南")
    print("="*50)

    common_issues = [
        {
            "問題": "ModuleNotFoundError",
            "原因": "Python路徑或模組導入問題",
            "解決方案": [
                "確認項目根目錄在Python路徑中",
                "檢查相對導入路徑是否正確",
                "確認所有依賴套件已安裝"
            ]
        },
        {
            "問題": "模型載入失敗",
            "原因": "模型文件不存在或損壞",
            "解決方案": [
                "執行 download_models.py 重新下載模型",
                "檢查磁盤空間是否足夠",
                "確認網絡連接正常"
            ]
        },
        {
            "問題": "OCR識別準確度低",
            "原因": "圖像品質或OCR參數設置問題",
            "解決方案": [
                "調整 OCR_MIN_CONFIDENCE 閾值",
                "啟用影像預處理 (deskew, CLAHE)",
                "嘗試不同的OCR引擎組合"
            ]
        },
        {
            "問題": "假資料生成效果不佳",
            "原因": "GPT-2模型或生成參數問題",
            "解決方案": [
                "調整 temperature 和 top_p 參數",
                "使用更大的GPT-2模型",
                "增加生成候選數量"
            ]
        }
    ]

    for issue in common_issues:
        print(f"\n問題: {issue['問題']}")
        print(f"原因: {issue['原因']}")
        print("解決方案:")
        for solution in issue['解決方案']:
            print(f"  • {solution}")


if __name__ == "__main__":
    # 基本使用演示
    main()

    print("\n" + "="*60)

    # 進階使用演示
    # advanced_usage_example()

    # 故障排除指南
    troubleshooting_guide()
