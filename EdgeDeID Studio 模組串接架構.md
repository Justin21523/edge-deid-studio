# EdgeDeID Studio 模組串接架構

## 完整處理流程

```
輸入文件/圖像
    ↓
[text_extractor.py] ── 文字抽取
    │
    ├─ PDF → pdfplumber + OCR備援
    ├─ DOCX → python-docx
    ├─ 圖像 → OCR處理
    └─ 其他格式處理
    ↓
[ocr.py] ── 圖像文字識別
    │
    ├─ 影像預處理 (去傾斜、增強、去噪)
    ├─ 版面分析 (LayoutLMv3)
    ├─ 多引擎OCR (Tesseract/EasyOCR/Ensemble)
    └─ 文字塊座標映射
    ↓
[position_mapper.py] ── 座標映射
    │
    └─ 字符索引 ↔ 邊界框映射
    ↓
[PII Detectors] ── 敏感資料偵測
    │
    ├─ [regex_detector.py] - 正則表達式偵測
    ├─ [bert_detector.py] - BERT NER模型
    ├─ [bert_onnx_detector.py] - ONNX加速推理
    └─ [composite.py] - 多偵測器聚合
    ↓
[replacer.py] ── 替換處理
    │
    ├─ 遮蔽模式 (MASK/REDACT)
    ├─ 假資料模式 (FAKE)
    └─ 部分遮蔽模式 (PARTIAL)
    ↓
[fake_provider.py] ── 假資料生成
    │
    ├─ GPT-2 語言模型生成
    ├─ Faker 結構化資料
    ├─ 中文本地化支援
    └─ 一致性快取機制
    ↓
[processor.py] ── 整合處理器
    │
    ├─ 流程編排
    ├─ 結果整合
    ├─ 視覺化標註
    └─ 輸出格式化
    ↓
最終去識別化結果
```

## 關鍵串接點說明

### 1. 文字抽取階段 (text_extractor.py → ocr.py)

```python
# text_extractor.py 中的核心邏輯
class SmartTextExtractor:
    def extract_text(self, file_path: str) -> DocumentLayout:
        if self._is_image_file(file_path):
            # 直接使用 OCR
            return self.ocr_adapter.process_image(file_path)
        elif file_path.endswith('.pdf'):
            # 嘗試原生抽取
            native_text = self._extract_pdf_native(file_path)
            if len(native_text.strip()) < OCR_THRESHOLD:
                # 回退到 OCR
                return self.ocr_adapter.process_pdf(file_path)
        # ... 其他格式處理
```

### 2. OCR處理階段 (ocr.py → position_mapper.py)

```python
# ocr.py 核心處理
class OCRAdapter:
    def process_image(self, image_path: str) -> DocumentLayout:
        # 1. 影像預處理
        processed_image = self._preprocess_image(image_path)

        # 2. 版面分析
        layout_blocks = self._analyze_layout(processed_image)

        # 3. OCR識別
        ocr_results = self._perform_ocr(processed_image, layout_blocks)

        # 4. 生成DocumentLayout with位置信息
        return self._create_document_layout(ocr_results)
```

### 3. PII偵測階段 (composite.py 統籌各偵測器)

```python
# composite.py 整合邏輯
class CompositeDetector:
    def detect(self, text: str) -> List[Entity]:
        all_entities = []

        # 1. 各偵測器並行執行
        for detector in self.detectors:
            entities = detector.detect(text)
            all_entities.extend(entities)

        # 2. 實體去重和優先級排序
        merged_entities = self._merge_entities(all_entities)

        # 3. 衝突解決
        final_entities = self._resolve_conflicts(merged_entities)

        return final_entities
```

### 4. 替換處理階段 (replacer.py ↔ fake_provider.py)

```python
# replacer.py 調用 fake_provider.py
class Replacer:
    def _create_fake_replacement(self, entity: Entity, original_value: str) -> str:
        # 根據實體類型選擇生成方法
        if entity.entity_type == 'PERSON':
            return self.fake_provider.generate_fake_name(
                original_value, preserve_format=self.preserve_format
            )
        elif entity.entity_type == 'PHONE':
            return self.fake_provider.generate_fake_phone(
                original_value, preserve_format=self.preserve_format
            )
        # ... 其他類型處理
```

### 5. 整合處理階段 (processor.py 統籌全流程)

```python
# processor.py 主要處理流程
class ImageDeidProcessor:
    def process_image(self, image_path: str) -> ProcessingResult:
        # 1. 文字抽取
        document_layout = self.text_extractor.extract_text(image_path)

        # 2. PII偵測
        entities = self.pii_detector.detect(document_layout.get_text())

        # 3. 座標映射
        positioned_entities = self.position_mapper.map_entities_to_positions(
            entities, document_layout
        )

        # 4. 替換處理
        replacement_result = self.replacer.replace_entities(
            document_layout.get_text(), positioned_entities
        )

        # 5. 視覺化和輸出
        return self._create_final_result(
            document_layout, replacement_result, positioned_entities
        )
```

## 配置文件整合

### config.py 全域參數控制

```python
# 核心配置參數
class Config:
    # OCR 相關
    OCR_THRESHOLD = 50  # PDF回退OCR門檻
    USE_LAYOUT_MODEL = True
    DESKEW_MIN_ANGLE = 0.5

    # PII偵測相關
    BERT_CONFIDENCE_THRESHOLD = 0.8
    USE_REGEX_DETECTOR = True
    USE_BERT_DETECTOR = True

    # 替換相關
    DEFAULT_REPLACEMENT_MODE = "MASK"
    PRESERVE_FORMAT = True
    USE_CONSISTENCY_CACHE = True

    # 模型路徑
    GPT2_MODEL_PATH = "models/gpt2"
    BERT_NER_MODEL_PATH = "models/ner"
    LAYOUT_MODEL_PATH = "models/layout"
```

## 關鍵數據流轉

### Entity 數據結構貫穿全流程

```python
@dataclass
class Entity:
    text: str           # 原始文字
    entity_type: str    # 實體類型 (PERSON, PHONE, EMAIL...)
    start: int          # 文字起始位置
    end: int            # 文字結束位置
    confidence: float   # 信心分數
    detector_name: str  # 偵測器名稱
    bbox: Optional[BoundingBox] = None  # 邊界框 (圖像處理時)
```

### DocumentLayout 版面結構

```python
@dataclass
class DocumentLayout:
    pages: List[PageLayout]     # 頁面列表
    metadata: Dict[str, Any]    # 元數據

    def get_text(self) -> str:
        """獲取全文"""
        return "\n".join(page.get_text() for page in self.pages)

    def get_text_blocks(self) -> List[TextBlock]:
        """獲取所有文字塊"""
        blocks = []
        for page in self.pages:
            blocks.extend(page.text_blocks)
        return blocks
```

## 模組依賴關係

### 核心依賴鏈

```
config.py (全域配置)
    ↓
base.py (Entity、PIIDetector基類)
    ↓
layout.py (DocumentLayout、PageLayout、TextBlock)
    ↓
position_mapper.py (TextPositionMapper)
    ↓
ocr.py (OCRAdapter) ← text_extractor.py (SmartTextExtractor)
    ↓
pii/各種偵測器 (regex_detector, bert_detector, composite)
    ↓
utils/fake_provider.py ← utils/replacer.py
    ↓
image_deid/processor.py (整合處理器)
    ↓
main.py (CLI介面)
```

### Import 路徑規範

```python
# 正確的 import 方式
from src.deid_pipeline.base import Entity, PIIDetector
from src.deid_pipeline.parser.layout import DocumentLayout, PageLayout
from src.deid_pipeline.parser.ocr import OCRAdapter
from src.deid_pipeline.parser.text_extractor import SmartTextExtractor
from src.deid_pipeline.pii.composite import CompositeDetector
from src.deid_pipeline.utils.replacer import Replacer, ReplacementMode
from src.deid_pipeline.utils.fake_provider import FakeProvider
from src.deid_pipeline.image_deid.processor import ImageDeidProcessor
```

## 實際使用流程

### 1. 系統初始化

```python
# 初始化各組件
def initialize_system():
    # 1. 載入配置
    config = Config()

    # 2. 初始化文字抽取器
    ocr_adapter = OCRAdapter(
        use_layout_model=config.USE_LAYOUT_MODEL,
        layout_model_path=config.LAYOUT_MODEL_PATH
    )
    text_extractor = SmartTextExtractor(ocr_adapter=ocr_adapter)

    # 3. 初始化PII偵測器
    detectors = []
    if config.USE_REGEX_DETECTOR:
        detectors.append(RegexDetector())
    if config.USE_BERT_DETECTOR:
        detectors.append(BertONNXNERDetector(config.BERT_NER_MODEL_PATH))

    composite_detector = CompositeDetector(detectors)

    # 4. 初始化替換系統
    fake_provider = FakeProvider(
        gpt2_model_path=config.GPT2_MODEL_PATH,
        chinese_model_path=config.GPT2_CHINESE_MODEL_PATH
    )
    replacer = Replacer(
        mode=ReplacementMode(config.DEFAULT_REPLACEMENT_MODE),
        fake_provider=fake_provider,
        preserve_format=config.PRESERVE_FORMAT
    )

    # 5. 初始化主處理器
    processor = ImageDeidProcessor(
        text_extractor=text_extractor,
        pii_detector=composite_detector,
        replacer=replacer
    )

    return processor
```

### 2. 單一文件處理

```python
def process_single_file(file_path: str, output_path: str):
    # 初始化系統
    processor = initialize_system()

    # 處理文件
    result = processor.process_image(file_path)

    # 保存結果
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.processed_text)

    # 輸出統計
    print(f"處理完成:")
    print(f"- 原始文字長度: {len(result.original_text)}")
    print(f"- 偵測到PII實體: {len(result.entities_found)}")
    print(f"- 替換實體數量: {len(result.entities_replaced)}")
    print(f"- 處理時間: {result.processing_time:.2f}s")
```

### 3. 批次處理

```python
def batch_process_files(input_dir: str, output_dir: str):
    processor = initialize_system()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 支援的文件格式
    supported_formats = ['.pdf', '.docx', '.txt', '.jpg', '.png', '.csv']

    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in supported_formats:
            try:
                print(f"處理文件: {file_path.name}")

                # 處理文件
                result = processor.process_image(str(file_path))

                # 保存結果
                output_file = output_path / f"{file_path.stem}_deid{file_path.suffix}"

                if file_path.suffix.lower() in ['.jpg', '.png']:
                    # 圖像文件保存標註版本
                    result.save_annotated_image(str(output_file))
                else:
                    # 文字文件保存清理後文字
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.processed_text)

                print(f"✓ 完成: {output_file}")

            except Exception as e:
                print(f"✗ 處理失敗 {file_path.name}: {e}")
```

## 性能優化建議

### 1. 模型載入優化

```python
# 使用單例模式避免重複載入模型
class ModelManager:
    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, model_type: str, model_path: str):
        if model_type not in self._models:
            if model_type == 'gpt2':
                self._models[model_type] = GPT2LMHeadModel.from_pretrained(model_path)
            elif model_type == 'bert_onnx':
                self._models[model_type] = ort.InferenceSession(model_path)
        return self._models[model_type]
```

### 2. 快取機制

```python
# 在 replacer.py 中實現多級快取
class CacheManager:
    def __init__(self):
        self.memory_cache = {}  # 記憶體快取
        self.disk_cache_path = "cache/replacements.json"  # 磁碟快取

    def get_replacement(self, key: str) -> Optional[str]:
        # 1. 檢查記憶體快取
        if key in self.memory_cache:
            return self.memory_cache[key]

        # 2. 檢查磁碟快取
        disk_value = self._load_from_disk(key)
        if disk_value:
            self.memory_cache[key] = disk_value
            return disk_value

        return None

    def set_replacement(self, key: str, value: str):
        self.memory_cache[key] = value
        self._save_to_disk(key, value)
```

### 3. 並行處理

```python
# 多執行緒處理多個文件
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def parallel_batch_process(file_paths: List[str], max_workers: int = 4):
    processor = initialize_system()

    def process_single(file_path: str):
        return processor.process_image(file_path)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single, file_paths))

    return results
```

## 錯誤處理和日誌

### 全局錯誤處理策略

```python
import logging
from typing import Optional

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edge_deid.log'),
        logging.StreamHandler()
    ]
)

class EdgeDeidError(Exception):
    """EdgeDeID 自定義異常基類"""
    pass

class ModelLoadError(EdgeDeidError):
    """模型載入錯誤"""
    pass

class ProcessingError(EdgeDeidError):
    """處理錯誤"""
    pass

# 在每個模組中添加錯誤處理
class RobustProcessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def safe_process(self, input_data, fallback_handler=None):
        try:
            return self._core_process(input_data)
        except Exception as e:
            self.logger.error(f"處理失敗: {e}", exc_info=True)
            if fallback_handler:
                return fallback_handler(input_data, e)
            raise ProcessingError(f"處理失敗: {e}") from e
```

## 測試和驗證

### 單元測試框架

```python
# tests/test_integration.py
import pytest
from src.deid_pipeline.image_deid.processor import ImageDeidProcessor

class TestIntegration:
    @pytest.fixture
    def processor(self):
        return initialize_system()

    def test_end_to_end_image_processing(self, processor):
        """測試端到端圖像處理"""
        test_image = "test_input/sample_document.png"
        result = processor.process_image(test_image)

        assert result.processed_text != result.original_text
        assert len(result.entities_found) > 0
        assert result.processing_time > 0

    def test_multiple_entity_types(self, processor):
        """測試多種實體類型偵測"""
        test_text = "我是張三，電話是0912345678，email是test@example.com"

        # 這裡需要調用適當的方法
        entities = processor.pii_detector.detect(test_text)

        entity_types = {e.entity_type for e in entities}
        assert 'PERSON' in entity_types
        assert 'PHONE' in entity_types
        assert 'EMAIL' in entity_types
```

這樣的架構設計確保了各模組間的清晰分工和有效協作，同時保持了系統的可擴展性和維護性。
