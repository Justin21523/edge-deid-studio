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
