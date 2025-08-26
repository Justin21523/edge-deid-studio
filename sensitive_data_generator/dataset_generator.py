# sensitive_data_generator/dataset_generator.py

# dataset_generator.py

import os
import time
import json
import random
from datetime import datetime

from .generators import (
    InsuranceGenerator,
    MedicalGenerator,
    FinancialGenerator,
    LegalGenerator,
    ConsentGenerator,
    EmployeeGenerator,
)
from .formatters import (
    InsuranceContractFormatter,
    MedicalRecordFormatter,
    FinancialReportFormatter,
    LegalAgreementFormatter,
    ConsentFormFormatter,
    EmployeeRecordFormatter,
)
from .advanced_file_writers import AdvancedFileWriter

class MultiFormatDatasetGenerator:
    """
    結合舊版 MultiFormatDatasetGenerator 與 新版動態 DatasetGenerator
    支援 contract/medical/financial 以及其他類型，
    並一次輸出 PDF/Word/Image/Excel/PPT/Text 等多種格式。
    """

    # 文件型別 → 預設要輸出的格式清單
    DOCUMENT_TEMPLATES = {
        "insurance_contract": {"formats": ["pdf", "word", "scanned"]},
        "medical_record":      {"formats": ["pdf", "word", "scanned"]},
        "financial_report":    {"formats": ["pdf", "word", "scanned", "excel", "ppt"]},
        "legal_agreement":     {"formats": ["pdf", "word", "scanned"]},
        "consent_form":        {"formats": ["pdf", "word", "scanned"]},
        "employee_record":     {"formats": ["pdf", "word", "scanned"]},
    }

    # 文件型別 → 對應的 Generator 類別
    GENERATOR_MAP = {
        "insurance_contract": InsuranceGenerator,
        "medical_record":      MedicalGenerator,
        "financial_report":    FinancialGenerator,
        "legal_agreement":     LegalGenerator,
        "consent_form":        ConsentGenerator,
        "employee_record":     EmployeeGenerator,
    }

    # 文件型別 → 對應的 Formatter 類別
    FORMATTER_MAP = {
        "insurance_contract": InsuranceContractFormatter,
        "medical_record":      MedicalRecordFormatter,
        "financial_report":    FinancialReportFormatter,
        "legal_agreement":     LegalAgreementFormatter,
        "consent_form":        ConsentFormFormatter,
        "employee_record":     EmployeeRecordFormatter,
    }

    # 簡單定義各 Format 用哪支 AdvancedFileWriter 方法
    WRITER_MAP = {
        "pdf":     AdvancedFileWriter.create_complex_pdf,
        "word":    AdvancedFileWriter.create_word_document,
        "scanned": AdvancedFileWriter.create_scanned_document,
        "excel":   AdvancedFileWriter.create_excel_spreadsheet,
        "ppt":     AdvancedFileWriter.create_powerpoint_presentation,
        # 若要純 text，也可以直接 open/write
    }

    def __init__(self, output_dir: str, count: int = 50,
                 document_types: list = None):
        """
        output_dir: 輸出總目錄
        count:      每一種文件型別要生成的檔案數
        document_types: 要包含的文件型別清單
        """
        self.output_dir = output_dir
        self.count = count
        # 若不指定，就只做三大類
        self.document_types = document_types or [
            "insurance_contract", "medical_record", "financial_report"
        ]

        # 建立各子資料夾
        self.sub_dirs = {
            fmt: os.path.join(output_dir, fmt)
            for fmt in ["pdf", "word", "scanned", "excel", "ppt"]
        }
        # 專屬類別目錄
        for dt in self.document_types:
            self.sub_dirs.setdefault(dt, os.path.join(output_dir, dt))

        # mkdir
        for path in self.sub_dirs.values():
            os.makedirs(path, exist_ok=True)

    def generate(self):
        """主流程：針對每個 document_type 生成多格式資料集"""
        dataset = []

        for doc_type in self.document_types:
            gen_cls = self.GENERATOR_MAP[doc_type]
            fmt_cls = self.FORMATTER_MAP[doc_type]
            formats = self.DOCUMENT_TEMPLATES[doc_type]["formats"]

            for idx in range(self.count):
                # 隨機 variant
                variants = gen_cls()._get_variants() \
                    if hasattr(gen_cls, "_get_variants") \
                    else gen_cls().variants
                variant = random.choice(variants)

                # 1) 產生原始資料
                raw_data = gen_cls().generate(variant=variant)

                # 2) 填充共通 metadata
                raw_data.update({
                    "document_id":       f"{doc_type[:3].upper()}-{time.time_ns()}",
                    "generation_date":   datetime.now().isoformat(),
                    "document_variant":  variant
                })

                # 3) 格式化
                formatted = fmt_cls().format(raw_data)

                # 4) 寫出多種格式
                item = {"id": f"{doc_type}_{idx+1}", "formats": []}
                for fmt in formats:
                    writer_fn = self.WRITER_MAP[fmt]
                    filename = f"{doc_type}_{variant}_{idx+1}"
                    # AdvancedFileWriter API 接受 (content, out_dir, filename)
                    out_path = writer_fn(
                        formatted,
                        self.sub_dirs[fmt],
                        f"{filename}.{fmt if fmt!='scanned' else 'png'}"
                    )
                    item["formats"].append({"format": fmt, "path": out_path})

                # 5) 同步留一份 .txt 原始內容
                txt_path = os.path.join(
                    self.sub_dirs[doc_type],
                    f"{doc_type}_{variant}_{idx+1}.txt"
                )
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(formatted)
                item["formats"].append({"format": "txt", "path": txt_path})

                dataset.append(item)

        # 最後把 metadata JSON 存起來
        meta_path = os.path.join(self.output_dir, "dataset_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"✅ 已在 `{self.output_dir}` 下生成 {len(dataset)} 個多格式文件")
        return dataset
