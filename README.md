# EdgeDeID Studio

EdgeDeID Studio is a real-time, on-device personal data anonymization toolkit that detects and redacts sensitive information (PII) from PDF documents, images, and tabular data within **150 ms**.

## ✨ Features

- 🔍 **NER + OCR PII Detection**: Identifies names, emails, addresses, ID numbers, and more.
- 🧠 **Generative AI Augmentation**: Replace redacted info with synthetic names, or generate summaries.
- 📄 **Document Support**: Works with PDF, image, and CSV/Excel files.
- ⚡ **Edge-Optimized**: Quantized ONNX models run on Qualcomm Copilot+ NPU with <150ms latency.
- 🛡️ **Privacy-First**: Everything runs locally. No data leaves the device.

## 🧰 Tech Stack

- **NER model**: `ckiplab/bert-base-chinese-ner`
- **Fake data generation**: `uer/gpt2-chinese-cluecorpussmall`
- **PDF/Image parsing**: `PyMuPDF`, `Pillow`, `pandas`
- **ONNX Inference**: `onnx`, `onnxruntime`, `onnxsim`
- **UI**: PySide6 (for graphical interface)

## 🗂️ Project Structure

## PII Models
### 🧰 [Azure AI](https://learn.microsoft.com/zh-tw/azure/ai-services/language-service/personally-identifiable-information/overview?source=recommendations&tabs=text-pii) 語言個人標識資訊 PII detection
Python ver: [Azure Python](https://learn.microsoft.com/zh-tw/azure/ai-services/language-service/personally-identifiable-information/quickstart?tabs=windows&pivots=programming-language-python)

- 可以參考使用，但是要收費 -> 免付費版本會有應用上的限制

### 🧰 [Better Data AI](https://huggingface.co/betterdataai/PII_DETECTION_MODEL)

- 不確定好不好用

```python
user_input = "Write an email to Julia indicating I won't be coming to office on the 29th of June"
new_prompt = prompt.format(classes="\n".join(classes_list) , text=user_input)
tokenized_input = tokenizer(new_prompt , return_tensors="pt").to(device)

output = model.generate(**tokenized_input , max_new_tokens=6000)
pii_classes = tokenizer.decode(output[0] , skip_special_tokens=True).split("The PII data are:\n")[1]

print(pii_classes)

##output
"""
<name> : ['Julia']
<date> : ['the 29th of June']
"""
```

### 🧰 [predidio](https://github.com/microsoft/presidio)
#### [Demo](https://huggingface.co/spaces/presidio/presidio_demo)

- Data Protection and De-identification SDK
- 效果佳

#### 難點
- 多種語言難一次偵測(除非直接使用多語 PII NER 模型偵測)
- Spacy 一次只能偵測一種語言 (需要多次呼叫 -> 效能 bad bad | 使用者端預先選擇 input file 的語言)

### 🧰 [Multilingual NER](https://huggingface.co/Babelscape/wikineural-multilingual-ner)
- mBERT multilingual language model
- model is trained on WikiNEuRal (Therefore, it might not generalize well to all textual genres (e.g. news))

### 🧰 [xlm-roberta-base-ner-hrl](https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl)
- based on a fine-tuned XLM-RoBERTa base model

### 🧰 [piiranha-v1-detect-personal-information](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information)
- open in Colab 可以直接實測
- 

下面彙整從最初到目前，我們在 **EdgeDeID Studio** 專案中所實作的全部功能、檔案結構與測試策略，並說明每個模組如何串接成「去識別化＋替換假資料」的完整流程，以及我們如何生成＆應用敏感假資料。

---

## 一、專案目錄總覽

```
/
├── configs/
│   └── regex_zh.yaml                   # 中文正則規則
│
├── models/                             # 原始 Hugging Face 模型快取
│   ├── ner/bert-ner-zh/
│   └── gpt2/
│
├── edge_models/
│   └── bert-ner-zh.onnx                # ONNX 格式 NER 模型
│
├── scripts/                            # 各種工具腳本
│   ├── download_models.py              # 一鍵下載 HF 模型
│   ├── run_automated_pipeline.py       # 自動化測試管線
│   ├── benchmark_formats.py            # 格式效能基準測試
│   └── validate_quality.py             # 去識別化品質驗證
│
├── examples/                           # 使用範例
│   ├── usage_examples.py               # 基本 & 批量資料生成示範
│   └── advanced_usage.py               # 進階使用範例
│
├── sensitive_data_generator/           # 假資料生成子系統
│   ├── __init__.py                     # 套件匯出介面
│   ├── config.py                       # 地區、街道、姓名、醫院等設定
│   ├── generators.py                   # 各類 PII Generator
│   ├── formatters.py                   # 基本段落 & 文件模板
│   ├── advanced_formatters.py          # 進階合約／醫療報告／財務報表模板
│   ├── file_writers.py                 # 基本 TXT/PDF/Image/CSV/JSON 輸出
│   ├── advanced_file_writers.py        # 進階 PDF/Word/Excel/PPT/掃描檔輸出
│   └── dataset_generator.py            # 一鍵產出多格式測試資料集
│
├── src/deid_pipeline/                  # 核心 De-ID Pipeline
│   ├── __init__.py                     # 匯出 DeidPipeline 類
│   ├── config.py                       # Pipeline 全域設定
│   ├── parser/                         # 檔案文字抽取
│   │   ├── ocr.py                      # EasyOCR singleton
│   │   └── text_extractor.py           # PDF/DOCX/Image → 純文字
│   ├── image_deid/                     # 影像去識別化
│   │   └── processor.py                # OCR→Detect→Replace→回寫圖片
│   ├── pii/                            # PII 偵測 & 假資料替換核心
│   │   ├── detectors/                  # 各種偵測器
│   │   │   ├── regex_detector.py
│   │   │   ├── spacy_detector.py
│   │   │   ├── bert_detector.py
│   │   │   ├── bert_onnx_detector.py
│   │   │   └── composite.py            # 多 detector 結果合併
│   │   └── utils/                      # 共用工具
│   │       ├── base.py                 # Entity, PIIDetector 抽象類
│   │       ├── fake_provider.py        # GPT-2 + Faker 假資料產生器
│   │       └── replacer.py             # 文本 & 事件記錄取代邏輯
│
└── tests/                              # 各層測試
    ├── test_data_factory.py            # Faker 測試資料產生
    ├── pii_test_suite.py               # Regex/BERT/Composite/Replacer 單元
    ├── test_detectors.py               # 多 detector 參數化測試
    ├── test_replacer.py                # 替換一致性測試
    ├── test_onnx_speed.py              # ONNX 延遲基準 (<25ms)
    ├── integration_test.py             # extract→detect→replace 整合測
    ├── performance_test.py             # 不同長度文本效能趨勢
    ├── end_to_end_test.py              # TXT/PDF/Image E2E 測試
    └── test_data_generator_integration.py  # 假資料生成器 + Pipeline 整合驗證
```

---

## 二、核心模組與功能

### 1. De-ID Pipeline (`src/deid_pipeline/`)

* **`config.py`**
  管理模型路徑、閾值、OCR 設定、Fake-data 參數、ONNX 開關等。
* **文字抽取 (`parser/`)**

  * `text_extractor.py`：PDF（`fitz`）、DOCX（`python-docx`）、影像（`EasyOCR`）→ 統一 `extract_text()`。
* **影像去識別 (`image_deid/processor.py`)**
  OCR → `get_detector()` 偵測 → `Replacer.replace()` → 塗黑或替換 → 回寫圖片。
* **PII 偵測 & 假資料替換 (`pii/`)**

  * **RegexDetector**：YAML 規則 → `re.finditer`。
  * **SpaCyDetector**：spaCy NER + regex 補正。
  * **BertDetector**、**BertONNXDetector**：Sliding window → Transformer 推論。
  * **Composite**：依 `ENTITY_PRIORITY` 整合多檢測器結果。
  * **FakeProvider**：GPT-2 + Faker fallback 生成假值。
  * **Replacer**：依 span 在原文替換或塗黑，並記錄事件。

整合成 `DeidPipeline.process(input)` → 回傳 `DeidResult(entities, output, report)`。

### Config.py 參數範例

```python
# src/deid_pipeline/config.py

# 1. 規則檔路徑
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CONFIGS_DIR    = PROJECT_ROOT / "configs"
REGEX_RULES_FILE = CONFIGS_DIR / "regex_zh.yaml"

def load_regex_rules(path: Path = REGEX_RULES_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

class Config:
    """全域設定中心：文字抽取／PII 偵測／假資料生成"""

    # 支援檔案類型
    SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".png", ".jpg"]

    # --- 文字抽取設定 ---
    OCR_ENABLED      = True
    OCR_THRESHOLD    = 50
    OCR_LANGUAGES    = ["ch_tra", "en"]

    # --- BERT 偵測設定 ---
    NER_MODEL_PATH          = os.getenv("NER_MODEL_PATH", PROJECT_ROOT / "models" / "ner")
    BERT_CONFIDENCE_THRESHOLD = 0.85
    MAX_SEQ_LENGTH          = 512
    WINDOW_STRIDE           = 0.5
    ENTITY_PRIORITY = {
        "TW_ID": 100,
        "PASSPORT": 95,
        "PHONE": 85,
        "EMAIL": 80,
        "NAME": 75,
        "ADDRESS": 70,
    }

    # --- Regex 規則 ---
    REGEX_PATTERNS = load_regex_rules()

    # --- 假資料生成 ---
    GPT2_MODEL_PATH   = os.getenv("GPT2_MODEL_PATH", PROJECT_ROOT / "models" / "gpt2")
    FAKER_LOCALE      = "zh_TW"
    FAKER_CACHE_SIZE  = 1000

    # --- ONNX Runtime 推論 ---
    USE_ONNX         = True
    ONNX_MODEL_PATH  = os.getenv("ONNX_MODEL_PATH", PROJECT_ROOT / "edge_models" / "bert-ner-zh.onnx")
    ONNX_PROVIDERS   = ["CPUExecutionProvider","CUDAExecutionProvider","NPUExecutionProvider"]

    # --- Logging & 環境旗標 ---
    ENVIRONMENT      = os.getenv("ENV", "local")
    LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_PROFILING = False
    USE_STUB         = False
````

> **說明**：
>
> * `OCR_*`：PDF 文字擷取的閾值與語言配置；
> * `NER_MODEL_PATH` 等：BERT 模型路徑與 sliding-window 參數；
> * `REGEX_PATTERNS`：載入 YAML 形式的 PII 正則；
> * `USE_ONNX`：切換到 ONNX Runtime；
> * 其餘為 Fake-data、Logging、環境控制旗標。


#### 1. Detector 組裝 (`detectors/__init__.py`)

```python
def get_detector(lang: str = "zh") -> CompositeDetector:
    config = Config()
    if lang == "zh" and not config.USE_STUB and MODEL_ZH.exists():
        return CompositeDetector(
            BertNERDetector(str(MODEL_ZH)),
            RegexDetector()
        )
    # 省略其他分支……
    else:
        return CompositeDetector(
            SpacyDetector(),
            RegexDetector(config_path="configs/regex_en.yaml")
        )
````

> **說明**：動態挑選 BERT/ONNX 或 SpaCy+Regex，並包成 CompositeDetector。

---

#### 2. Entity 定義 (`utils/base.py`)

```python
class Entity(TypedDict):
    span: Tuple[int, int]     # 原文中字元位置 (start, end)
    type: PII_TYPES           # PII 類型，例如 NAME、ID、PHONE…
    score: float              # 偵測信心值
    source: str               # 偵測來源，如 "bert", "regex", "spacy"
```

> **說明**：用 TypedDict 定義可序列化的 PII 實體結構，統一流轉格式。

---

#### 3. 管線入口 (`src/deid_pipeline/__init__.py`)

```python
class DeidPipeline:
    def __init__(self, language: str = "zh"):
        self.detector = get_detector(language)
        self.replacer = Replacer()
        self.ocr_proc = ImageDeidProcessor(lang=language)

    def process(self, input_path: str, output_mode: str = "replacement"):
        # 1. 文字或影像抽取
        suffix = input_path.lower().rsplit(".", 1)[-1]
        if suffix in ("txt", "docx", "pdf"):
            text, _ = extract_text(input_path)
        else:
            ocr_res = self.ocr_proc.process_image(input_path)
            text = ocr_res["original_text"]

        # 2. 偵測
        entities = self.detector.detect(text)

        # 3. 替換或塗黑
        clean_text, events = self.replacer.replace(text, entities)
        return DeidResult(entities=entities, text=clean_text)
```

> **說明**：整合抽取→偵測→替換三大步驟，對外提供一致化介面。

---

### 2. 假資料生成子系統 (`sensitive_data_generator/`)

#### a. 基本生成

* **`config.py`**：台灣縣市、街道、姓氏、名字、醫院、專科清單。
* **`generators.py`**：

  * `generate_tw_id()`, `generate_tw_phone()`, `generate_tw_address()`, `generate_tw_name()`…
  * `generate_random_pii()` 隨機挑選一種 PII。
* **`formatters.py`**：

  * `generate_paragraph()`：自然語言段落模板，內嵌 PII、可調密度。
  * `generate_medical_record()`, `generate_financial_document()`, `generate_random_document()`。

#### b. 進階模板

* **`advanced_formatters.py`**：

  * `generate_contract_document()`：合約書範本。
  * `generate_medical_report()`：圖表引用的醫療報告段落。
  * `generate_financial_statement()`：Markdown 風格財務報表。

#### c. 檔案輸出

* **`file_writers.py`**：
  TXT、簡單 PDF、模擬掃描圖片 (PIL)、CSV、JSON。
* **`advanced_file_writers.py`**：

  * **ReportLab**：複雜 PDF（標題、表格、圖表）。
  * **python-docx**：Word（標題、表格、圖片、頁尾）。
  * **xlsxwriter**：Excel（格式化 + 圖表）。
  * **python-pptx**：PPT（投影片、表格、圖片）。
  * **PIL**：掃描文件模擬（雜訊、印章、簽名）。

#### d. 多格式資料集生成

* **`dataset_generator.py`**：
  `MultiFormatDatasetGenerator.generate_full_dataset(output_dir, num_items)`：

  1. 在各子目錄（pdf/word/excel/ppt/scanned/contracts/medical/financial）產出對應檔案。
  2. 同步儲存純文字版。
  3. 匯出 `dataset_metadata.json`，記錄每筆的格式、檔案路徑與部分內容。

---

## 三、測試程式碼 & 驗證項目

| 測試檔案                                 | 測試內容                                           |
| ------------------------------------ | ---------------------------------------------- |
| `test_data_factory.py`               | 驗證 Faker 產生資料集功能                               |
| `pii_test_suite.py`                  | Regex/BERT/Composite/Replacer 單元測試             |
| `test_detectors.py`                  | 多 detector 參數化 correctness                     |
| `test_replacer.py`                   | 相同原始字串替換一致性                                    |
| `test_onnx_speed.py`                 | ONNX 模式延遲 < 25 ms                              |
| `integration_test.py`                | `extract→detect→replace` 整合流程                  |
| `performance_test.py`                | 不同文本長度（1k/5k/10k/20k）效能基準                      |
| `end_to_end_test.py`                 | TXT/PDF/Image E2E 測試，準確度 ≥ 80%                 |
| `test_data_generator_integration.py` | 假資料生成器輸出與 `DeidPipeline` 整合，偵測率 ≥ 95%、一致性 100% |

### 測試目的

1. **功能正確性**：各 Detector、Replacer、Parser 單元輸出符合預期。
2. **整合流程**：Pipeline 從各格式抽取、PII 偵測到替換不漏讀、不破壞格式。
3. **效能基準**：ONNX vs PyTorch 推論速率；不同文本長度延遲。
4. **端到端（E2E）**：含影像 OCR → 偵測 → 替換，全面驗證。
5. **生成器驗證**：自動產生的假資料，必須能被 Pipeline 偵測，且替換一致。

---

## 四、敏感假資料生成與後續應用

1. **生成**：

   * 呼叫 `PIIGenerator` 系列方法產生單一 PII。
   * 透過 `DataFormatter`／`AdvancedDataFormatter` 把 PII 嵌入全文件文本或段落。
   * 再由 `AdvancedFileWriter`／`FileWriter` 輸出多種格式檔案。

2. **應用範例**：

   * 在 CI/CD 中先行產生 100+ 檔案，放到 `test_dataset/`。
   * 自動化測試管線 `run_automated_pipeline.py` → 驗證每個檔案 PII 偵測率、處理時間。
   * `benchmark_formats.py` → 比較 PDF、DOCX、XLSX、PNG 各自平均/最慢/最快處理時間。
   * `validate_quality.py` → 驗證原始 PIIs 是否全被移除，並檢查格式保留情況。

---

### PII 偵測器模組說明

#### `processor.py`  
路徑：`src/deid_pipeline/image_deid/processor.py`  
**功能定位**  
- 類別：`ImageDeidProcessor`  
- 負責：將影像 OCR → PII 偵測 → 替換／遮蔽 → 回傳含原文、清理後文字、偵測結果、事件與耗時  

**實作原理**  
1. 用 OpenCV 讀檔  
2. 透過 EasyOCR (singleton) 抽文字 `(bbox, text, conf)`  
3. 合併文字 → `original_text`  
4. 呼叫複合偵測器 `self.detector.detect(…)`  
5. 用 `self.replacer.replace(…)` 套上假資料或黑框  
6. 回傳所有中間結果與耗時  

---

#### `ocr.py`  
路徑：`src/deid_pipeline/parser/ocr.py`  
**功能定位**  
- 函式：`get_ocr_reader(langs)`  
- 負責：單例管理 EasyOCR Reader，預設讀取 `Config.OCR_LANGUAGES`，避免重複初始化  

**實作原理**  
```python
if _OCR_READER is None:
    _OCR_READER = easyocr.Reader(langs, gpu=False)
return _OCR_READER
````

* 單例模式節省模型載入時間
* 語言清單由 Config 控制

---

#### `text_extractor.py`

路徑：`src/deid_pipeline/parser/text_extractor.py`
**功能定位**

* 函式：`extract_text(file_path, ocr_fallback=True)`
* 負責：從多種格式（`.txt`、`.docx`、`.html`、`.pdf`）提取文字並回傳 offset map

**實作原理**

1. 文字／Word／HTML → 直讀全文 + 建立 char→(page, bbox) map
2. PDF → 用 `fitz` 抽 blocks，若文字過少(`len<Config.OCR_THRESHOLD`) → OCR fallback
3. 回傳 `(full_text, offset_map)`

---

### PII 偵測器系列

#### `spacy_detector.py`

路徑：`src/deid_pipeline/pii/detectors/legacy/spacy_detector.py`
**功能定位**

* SpaCy NER + Regex 雙刀流

**實作原理**

1. `nlp = spacy.load(...)` → `doc.ents`
2. 篩選 `SPACY_TO_PII_TYPE`
3. `Entity(..., score=0.99, source="spacy")`
4. 加入 `Config.REGEX_PATTERNS` 正則匹配 results
5. `_resolve_conflicts(...)` 保留最高分或優先級

---

#### `regex_detector.py`

路徑：`src/deid_pipeline/pii/detectors/regex_detector.py`
**功能定位**

* 單純用正則 `re.finditer` 掃 PII

**實作原理**

```python
for type, patterns in Config.REGEX_PATTERNS.items():
    for pat in patterns:
        for m in re.compile(pat).finditer(text):
            yield Entity(span=(m.start(), m.end()), type=type, score=1.0, source="regex")
```

---

#### `bert_detector.py`

路徑：`src/deid_pipeline/pii/detectors/bert_detector.py`
**功能定位**

* PyTorch Transformers BERT Token Classification

**實作原理**

1. `__init__`載入 ONNX 或 PyTorch 模型 + tokenizer
2. `detect(text)` → sliding window 切塊
3. 每段做推論 → 回傳 token label
4. `_merge_entities(...)` 去重合、依 `ENTITY_PRIORITY` 保留

---

#### `bert_onnx_detector.py`

路徑：`src/deid_pipeline/pii/detectors/bert_onnx_detector.py`
**功能定位**

* ONNX Runtime 加速版 BERT 偵測

**差異**

* 模型載入改用 `ORTModelForTokenClassification.from_pretrained(...)`
* 推論改成 `session.run(...)`

---

#### `composite.py`

路徑：`src/deid_pipeline/pii/detectors/composite.py`
**功能定位**

* 將前述所有偵測器結果「parallel 執行 → 合併去重」

**實作原理**

```python
all_ents = []
for det in self.detectors:
    all_ents.extend(det.detect(text))
return self._resolve_conflicts(all_ents)
```

* 依 `ENTITY_PRIORITY` 與 score 決定最終保留

### 偵測器與工具模組說明

#### `regex_detector.py`  
路徑：`src/deid_pipeline/pii/detectors/regex_detector.py`  
- **功能**：動態載入 `configs/regex_zh.yaml` 中的多個正則規則，對文字做全文掃描，回傳所有命中的 PII Entity  
- **實作要點**：  
  1. `load_rules()` 用 `os.path.getmtime` 檢查檔案更新並重載  
  2. 支援 `"IGNORECASE|MULTILINE"` 等多 flag 字串解析  
  3. `detect(text)` → `for (type,pattern) in rules: pattern.finditer(text)` → `Entity(span, type, score=1.0, source="regex")`

---

#### `__init__.py` (detectors)  
路徑：`src/deid_pipeline/pii/detectors/__init__.py`  
- **功能**：集中引入各 Detector 並實作 `get_detector(lang)`  
- **選擇邏輯**：  
  1. 根據語言 (`zh`/`en`)  
  2. `Config.USE_STUB` 開關  
  3. 若啟用 ONNX，且模型存在 → 回傳 ONNX + Regex  
  4. 否則回傳 PyTorch BERT + Regex  
  5. `CompositeDetector` 負責多檢測器合併與去衝突

---

#### `config.py`  
路徑：`src/deid_pipeline/config.py`  
- **功能**：全域設定中心  
- **主要設定**：  
  - Regex 規則檔路徑、`OCR_LANGUAGES`、`OCR_THRESHOLD`  
  - BERT：`NER_MODEL_PATH`, `MAX_SEQ_LENGTH`, `WINDOW_STRIDE`, `ENTITY_PRIORITY`  
  - ONNX：`USE_ONNX`, `ONNX_MODEL_PATH`, `ONNX_PROVIDERS`  
  - Fake-data：`GPT2_MODEL_PATH`, `FAKER_LOCALE`  
  - 管線旗標：`USE_STUB`, `ENABLE_PROFILING`, `LOG_LEVEL`  

---

#### `fake_provider.py`  
路徑：`src/deid_pipeline/pii/utils/fake_provider.py`  
- **功能**：混合 GPT-2 + Faker 的 PII 假資料產生  
- **實作要點**：  
  1. `GPT2Provider.generate(prompt)` → 失敗則  
  2. `Faker("zh_TW")` fallback  
  3. 內部 cache 避免重複生成同一原始字串

---

#### `replacer.py`  
路徑：`src/deid_pipeline/pii/utils/replacer.py`  
- **功能**：根據 `Entity.span` 有序替換或回傳遮黑座標  
- **實作要點**：  
  1. `entities` 先按 `start` 排序  
  2. 滑動拼接新字串，更新 `offset`  
  3. 支援 `"replace"` 與 `"black"` 模式  
  4. `dumps(events)` → JSON

---

#### 檔案串接

在 `src/deid_pipeline/pii/detectors/__init__.py` 中：

```python
from .spacy_detector import SpacyDetector
from .regex_detector import RegexDetector
from .bert_detector import BertNERDetector
from .bert_onnx_detector import BertONNXNERDetector
from .composite import CompositeDetector

def get_detector(lang="zh"):
    # 根據 Config.USE_ONNX / USE_STUB 組成 CompositeDetector(...)
    return CompositeDetector(...)
```

---  


### 🔐 sensitive_data_generator

這個子模組負責「合成」多格式、含敏感資料的假測試文件，供 De-ID pipeline 測試與 benchmark。

#### 2.1 `__init__.py`

```python
from .config import *
from .generators import PIIGenerator
from .formatters import DataFormatter
from .advanced_formatters import AdvancedDataFormatter
from .file_writers import FileWriter
from .advanced_file_writers import AdvancedFileWriter
from .dataset_generator import MultiFormatDatasetGenerator

__all__ = [
  "PIIGenerator", "DataFormatter", "FileWriter",
  "AdvancedDataFormatter","AdvancedFileWriter","MultiFormatDatasetGenerator"
]
````

* **功能**：把模組裡的核心類別一次導出 (`__all__`)，提供上層 `import sensitive_data_generator` 就能拿到產生器、格式器、檔案輸出等所有工具。

#### 2.2 `advanced_file_writers.py`

```python
class AdvancedFileWriter:
    """進階多格式檔案輸出工具"""

    @staticmethod
    def create_complex_pdf(content, output_dir, filename=None, include_charts=True):
        # 1. 確保目錄存在
        os.makedirs(output_dir, exist_ok=True)
        # 2. 建立 ReportLab PDF 文件
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # 3. 加標題與正文
        title = Paragraph("機密文件 – 個人資料報告", styles['Heading1'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        pii_para = Paragraph(content, styles['BodyText'])
        elements.append(pii_para)
        elements.append(Spacer(1, 12))

        # 4. 加表格（示範插入 4 欄：姓名、ID、電話、地址）
        table_data = [
          ['項目','原始資料','備註'],
          ['姓名', PIIGenerator.generate_tw_name(), '測試用虛擬姓名'],
          ['身分證', PIIGenerator.generate_tw_id(), '測試用虛擬ID'],
          ['電話', PIIGenerator.generate_tw_phone(), '測試用虛擬電話'],
          ['地址', PIIGenerator.generate_tw_address(), '測試用虛擬地址']
        ]
        table = Table(table_data, colWidths=[1.5*inch,3*inch,2.5*inch])
        table.setStyle(TableStyle([...]))
        elements.append(table)
        elements.append(Spacer(1, 24))

        # 5. 可選：插入假圖表，圖用 PIL+matplotlib 生成
        if include_charts:
            chart_img = AdvancedFileWriter.generate_fake_chart()
            elements.append(RLImage(chart_img, width=5*inch, height=3*inch))
            elements.append(Paragraph("圖1：測試資料分佈圖", styles['Italic']))

        # 6. 寫出 PDF
        doc.build(elements)
        return filepath
```

* **功能拆解**

  1. **目錄檢查**：`os.makedirs(...)`
  2. **PDF**：使用 ReportLab `SimpleDocTemplate` + `Paragraph`＋`Table`＋`Spacer`
  3. **假資料表格**：`PIIGenerator` 隨機生成姓名、ID、電話、地址
  4. **假圖表**：呼叫 `generate_fake_chart()` → 隨機產生 bar/line/pie 圖
  5. **匯出**：回傳完整檔案路徑

```python
    @staticmethod
    def generate_fake_chart():
        """生成 Bar/Line/Pie 假圖表"""
        plt.figure(figsize=(8,5))
        kind = random.choice(['bar','line','pie'])
        if kind=='bar':
            labels = ['A部門','B部門','C部門','D部門']
            values = np.random.randint(100,500,size=4)
            plt.bar(labels, values)
            plt.title('部門業績比較')
        elif kind=='line':
            x = np.arange(1,11)
            y = np.random.rand(10)*100
            plt.plot(x,y,marker='o')
            plt.title('月度趨勢分析')
        else:
            labels = ['類別A','類別B','類別C','類別D']
            sizes = np.random.randint(15,40,size=4)
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            plt.title('類別分佈圖')
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        return buf
```

* **功能**：用 matplotlib 隨機選擇圖表類型、生成數據後輸出到 `BytesIO`，讓上層 PDF/Word/PPTX 都可以直接插圖。

> **後續**：`create_word_document`、`create_powerpoint_presentation`、`create_excel_spreadsheet`、`create_scanned_document` 都採相同拆分：
>
> * **Word** → `python-docx`：`Document()`、`add_heading`、`add_table`、`add_picture`
> * **PPTX** → `python-pptx`：`Presentation()`、`slides.add_slide()`、`shapes.add_table()`、`shapes.add_picture()`
> * **Excel** → `pandas.DataFrame` + `ExcelWriter(engine='xlsxwriter')`；設定 header 格式、欄寬、數值格式
> * **掃描檔** → `PIL.ImageDraw`：畫背景噪點、文字、簽章、簽名，模擬掃描品質

下面示範如何把 **`advanced_formatters.py`**、**`config.py`**、**`dataset_generator.py`** 也同樣補到文件裡，並說明每個區塊的功能與目的。


#### 2.3 `advanced_formatters.py`

```python
class AdvancedDataFormatter:
    """進階資料格式化生成器"""

    @staticmethod
    def generate_contract_document():
        """
        產生一份合約合約範本（含虛擬當事人資料）：
        - parties: 隨機產生甲乙雙方姓名、身分證、地址、簽訂日期
        - contract: 填入各條款樣板（目的、期限、報酬、保密、管轄法院等）
        """
        parties = {
          "甲方": PIIGenerator.generate_tw_name(),
          "乙方": PIIGenerator.generate_tw_name(),
          "甲方身分證": PIIGenerator.generate_tw_id(),
          "乙方身分證": PIIGenerator.generate_tw_id(),
          "甲方地址": PIIGenerator.generate_tw_address(),
          "乙方地址": PIIGenerator.generate_tw_address(),
          "簽約日期": (datetime.now() - timedelta(days=random.randint(1,365)))\
             .strftime("%Y年%m月%d日")
        }
        contract = f"""
        合約書

        立合約當事人：
        甲方：{parties['甲方']}（身分證號：{parties['甲方身分證']}）
        ...
        第六條 管轄法院  
        因本合約發生之爭議，雙方同意以台灣台北地方法院為第一審管轄法院。

        中華民國 {parties['簽約日期']}
        """
        return contract
````

* **功能**：用 `PIIGenerator` 隨機填入「合約」所需關鍵欄位，並透過多行字串模板（f-string）組成完整合約範本。

```python
    @staticmethod
    def generate_medical_report():
        """
        生成詳細醫療報告文本（含虛擬病人資料 + 虛擬檢查數據）：
        - patient: 隨機姓名、ID、出生、電話、地址、病歷號
        - test_results: 血壓、心率、血糖、膽固醇等
        - report: f-string 填入醫院名稱、各節標題（病史、診斷、檢驗、影像、處方、醫囑）
        """
```

* **功能**：同樣用 f-string + `HOSPITALS` 列表隨機挑選醫院，組出可直接貼檔的醫療報告模板。

---

#### 2.4 `config.py`

```python
# 台灣地區常用參考資料，供 Formatter/Generator 使用
TAIWAN_LOCATIONS = {
  "北部": ["台北市","新北市","基隆市",...],
  "中部": ["台中市","彰化縣",...],
  ...
}

STREET_NAMES = ["中山","中正","光復",...]
SURNAMES     = ["陳","林","張",...]
GIVEN_NAMES  = ["怡君","志明","雅婷",...]
HOSPITALS    = ["台大醫院","長庚紀念醫院",...]
MEDICAL_SPECIALTIES = ["內科","外科","兒科",...]
```

* **功能**：把所有可隨機選用的地名、街道、姓名、醫院、科別等列表集中管理，方便 Formatter 呼叫。

---

#### 2.5 `dataset_generator.py`

```python
class MultiFormatDatasetGenerator:
    """多格式敏感資料集生成器"""

    @staticmethod
    def generate_full_dataset(output_dir, num_items=50):
        """
        一次生產多種格式（pdf、word、image、excel、ppt、contracts、medical、financial…）  
        - 建立子資料夾：pdf/、word/、scanned/、excel/、ppt/、contracts/、medical/、financial/  
        - 逐筆循環：隨機選 contract/medical/financial，呼叫 AdvancedDataFormatter 產文本  
        - 呼叫 AdvancedFileWriter 輸出對應格式檔案並紀錄路徑  
        - 最後匯出 metadata.json，包含每筆的格式清單與檔案位置
        """
        # 建目錄、初始化 dataset list…
        sub_dirs = {…}
        for i in range(num_items):
          doc_type = random.choice(["contract","medical","financial"])
          if doc_type=="contract":
            content = AdvancedDataFormatter.generate_contract_document()
          elif doc_type=="medical":
            content = AdvancedDataFormatter.generate_medical_report()
          else:
            content = AdvancedDataFormatter.generate_financial_statement()

          pdf_path = AdvancedFileWriter.create_complex_pdf(content, sub_dirs["pdf"], f"{doc_type}_{i+1}.pdf")
          item["formats"].append({"format":"pdf","path":pdf_path})

          # …同理呼叫 create_word_document、create_scanned_document
          # 若 financial 額外呼叫 create_excel_spreadsheet、create_powerpoint_presentation

          # 寫 content .txt、dataset.append(item)
        # 寫出 dataset_metadata.json
```

* **功能**：整合以上 Formatter + FileWriter，批次生產多格式測試集並輸出 metadata，便於後續自動化測試與 benchmark。

下面示範如何把 **`file_writers.py`**、**`formatters.py`**、**`generators.py`** 也加入說明，流程與先前一致：

#### 2.6 `file_writers.py`

```python
class FileWriter:
    """檔案輸出工具"""

    @staticmethod
    def write_text_file(content, output_dir, filename=None):
        """
        將文字內容寫入 .txt 檔
        - 自動建立資料夾
        - 若未指定 filename，則用 timestamp 命名
        - 回傳檔案完整路徑
        """
        ...

    @staticmethod
    def write_pdf_file(content, output_dir, filename=None):
        """
        將文字內容寫入 PDF
        - 使用 fpdf 套件
        - 支援多行文字排版（multi_cell）
        - 回傳檔案完整路徑
        """
        ...

    @staticmethod
    def write_csv_file(rows, output_dir, filename=None):
        """
        將 list-of-dict 寫成 CSV
        - 自動建立資料夾
        - 依 dict keys 作為欄位
        """
        ...
````

* **目的**：提供最基本的「文字 / PDF / CSV」檔案輸出能力，供上層 generator 輕鬆呼叫。

#### 2.7 `formatters.py`

```python
class DataFormatter:
    """敏感資料段落 & 文件範本生成器"""

    @staticmethod
    def generate_paragraph(min_sentences=3, max_sentences=8, pii_density=0.3):
        """
        用多種句型範本隨機拼出一段文字，並依照 pii_density 插入 PII
        - sentence_templates: 多種含佔位符 {NAME}/{PHONE}/{ADDRESS}… 的句子
        - 隨機決定要插幾句、每句是否要替換成 PII
        """
        ...

    @staticmethod
    def generate_medical_record():
        """
        生成完整醫療紀錄字串
        - 基本資訊（姓名/性別/出生/身分證/電話/地址/病歷號）
        - 就診資訊（日期/醫院/科別/醫師）
        - 診斷與處方、用藥建議
        """
        ...

    @staticmethod
    def generate_financial_document():
        """
        生成財務報表文字
        - 客戶基本資料（姓名/ID/聯絡/帳號/信用卡）
        - 隨機 3～10 筆交易記錄
        - 計算總餘額、支出統計
        """
        ...
```

* **目的**：將原始 PII 生成器（`PIIGenerator`）轉成可貼文件的自然段落或完整文件範本。

#### 2.8 `generators.py`

```python
class PIIGenerator:
    """繁體中文各類 PII 隨機生成器"""

    @staticmethod
    def generate_tw_id():
        """符合規則的臺灣身分證字號（含檢核碼）"""
        ...

    @staticmethod
    def generate_tw_phone():
        """臺灣手機號碼（0912-345-678 或 0912345678）"""
        ...

    @staticmethod
    def generate_tw_address():
        """臺灣地址：隨機區域 + 隨機街道 + 門牌 + 樓層"""
        ...

    @staticmethod
    def generate_tw_name():
        """隨機挑選常見姓氏 + 名字（有 30% 機率雙名）"""
        ...

    @staticmethod
    def generate_medical_record():
        """僅回傳「病歷號」格式，供範本插入"""
        ...

    @staticmethod
    def generate_credit_card():
        """模擬信用卡卡號（16 碼）"""
        ...

    ...（其他如 email、passport、license_plate、health_insurance、random_pii 等）...
```

* **目的**：低階 PII API，專注「產生一則」各種敏感欄位值，所有上層 Formatter / FileWriter / DatasetGenerator 都建構在它之上。


---

### 🛠️ Scripts utilities

### 1. `benchmark_formats.py` — 格式效能基準測試
```python
from deid_pipeline import DeidPipeline
def benchmark_formats(dataset_dir, formats=["pdf","docx","xlsx","png"]):
    pipeline = DeidPipeline(language="zh")
    for fmt in formats:
        fmt_files = [f for f in os.listdir(dataset_dir) if f.endswith(fmt)]
        # 每種格式只測 10 個檔案
        for file in fmt_files[:10]:
            start = time.time()
            pipeline.process(os.path.join(dataset_dir, file))
            processing_times.append(time.time()-start)
````

* **功能**：對指定資料夾中，各格式前10個檔案做去識別化，收集執行時間。
* **用途**：量化不同檔案格式（PDF、Word、Excel、PNG）在去識別化流程中的平均／最小／最大處理時間，幫助調優與資源規劃。

---

### 2. `download_models.py` — 模型預下載

```python
MODELS = {
  "ner_zh": ("ckiplab/bert-base-chinese-ner", "models/ner/bert-ner-zh"),
  "gpt2_base": ("gpt2", "models/gpt2")
}
for name, (repo_id, target) in MODELS.items():
    # Transformers 下載 GPT-2
    if name=="gpt2_base" and not (Path(target)/"pytorch_model.bin").exists():
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        model.save_pretrained(target); tokenizer.save_pretrained(target)
    # HF Hub snapshot 下載 NER
    elif not Path(target).exists():
        snapshot_download(repo_id, local_dir=target)
```

* **功能**：自動從 HuggingFace 及 Transformers 下載、快照保存 BERT-NER 與 GPT-2 模型到 `models/`。
* **用途**：確保團隊一鍵執行時已具備本地模型，避免首次運行時手動下載失敗。

---

### 3. `run_automated_pipeline.py` — 自動化測試管線

```python
from deid_pipeline import DeidPipeline
def run_automated_test_pipeline(dataset_dir):
    pipeline = DeidPipeline(language="zh")
    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            res = pipeline.process(os.path.join(root, fn))
            results.append({
                "file": fn,
                "format": fn.split(".")[-1],
                "pii_count": len(res.entities),
                "processing_time": res.processing_time
            })
    json.dump(results, open("pipeline_results.json","w"), ensure_ascii=False, indent=2)
```

* **功能**：遞迴遍歷資料集資料夾，對每支檔案呼叫 `DeidPipeline.process()`，並把 PII 偵測數、執行時間輸出成 JSON。
* **用途**：快速檢視整批測試資料的去識別化成效，方便生成報表或上傳 CI。

---

### 4. `validate_quality.py` — 去識別化品質驗證

```python
def validate_deidentification_quality(original_dir, processed_dir):
    for orig in os.listdir(original_dir):
        proc = os.path.join(processed_dir, orig)
        orig_text = open(os.path.join(original_dir,orig)).read()
        proc_text = open(proc).read()
        # 檢查是否移除所有 PII
        for label in ["身分證","電話","地址","病歷號"]:
            if label in orig_text and label in proc_text:
                pii_removed=False
        quality_report.append({...})
    # 計算成功率
    pii_success = sum(r["pii_removed"] for r in quality_report)/len(quality_report)
    print(f"PII Removal Success: {pii_success:.2%}")
```

* **功能**：逐一比對原檔與處理後檔，驗證「所有標註的 PII」確實未出現在去識別化結果中，同時可留待擴充「表格、圖表完整性檢查」。
* **用途**：在 CICD 流程中自動確認去識別化質量指標（PII 移除率、格式保留率）。

---
