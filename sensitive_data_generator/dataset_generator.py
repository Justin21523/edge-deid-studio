# sensitive_data_generator/dataset_generator.py

import os
import random
import json
from datetime import datetime
from .advanced_file_writers import AdvancedFileWriter
from .advanced_formatters import AdvancedDataFormatter

class MultiFormatDatasetGenerator:
    """多格式敏感資料集生成器"""

    @staticmethod
    def generate_full_dataset(output_dir, num_items=50):
        """生成完整多格式測試資料集"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 創建子目錄
        sub_dirs = {
            "pdf": os.path.join(output_dir, "pdf"),
            "word": os.path.join(output_dir, "word"),
            "excel": os.path.join(output_dir, "excel"),
            "ppt": os.path.join(output_dir, "ppt"),
            "scanned": os.path.join(output_dir, "scanned"),
            "contracts": os.path.join(output_dir, "contracts"),
            "medical": os.path.join(output_dir, "medical"),
            "financial": os.path.join(output_dir, "financial")
        }

        for path in sub_dirs.values():
            os.makedirs(path, exist_ok=True)

        # 生成資料集
        dataset = []

        for i in range(num_items):
            item = {
                "id": i+1,
                "formats": [],
                "metadata": {}
            }

            # 隨機選擇文件類型
            doc_type = random.choice(["contract", "medical", "financial"])

            if doc_type == "contract":
                content = AdvancedDataFormatter.generate_contract_document()
                doc_category = "contracts"
            elif doc_type == "medical":
                content = AdvancedDataFormatter.generate_medical_report()
                doc_category = "medical"
            else:  # financial
                content = AdvancedDataFormatter.generate_financial_statement()
                doc_category = "financial"

            item["metadata"]["type"] = doc_type
            item["metadata"]["content"] = content[:500] + "..."  # 儲存部分內容

            # 生成PDF版本
            pdf_path = AdvancedFileWriter.create_complex_pdf(
                content,
                sub_dirs["pdf"],
                f"{doc_type}_doc_{i+1}.pdf"
            )
            item["formats"].append({"format": "pdf", "path": pdf_path})

            # 生成Word版本
            word_path = AdvancedFileWriter.create_word_document(
                content,
                sub_dirs["word"],
                f"{doc_type}_doc_{i+1}.docx"
            )
            item["formats"].append({"format": "word", "path": word_path})

            # 生成掃描版本
            scanned_path = AdvancedFileWriter.create_scanned_document(
                content,
                sub_dirs["scanned"],
                f"{doc_type}_doc_{i+1}.png"
            )
            item["formats"].append({"format": "image", "path": scanned_path})

            # 特定類型額外格式
            if doc_type == "financial":
                excel_path = AdvancedFileWriter.create_excel_spreadsheet(
                    sub_dirs["excel"],
                    f"financial_data_{i+1}.xlsx"
                )
                item["formats"].append({"format": "excel", "path": excel_path})

                ppt_path = AdvancedFileWriter.create_powerpoint_presentation(
                    sub_dirs["ppt"],
                    f"financial_report_{i+1}.pptx"
                )
                item["formats"].append({"format": "ppt", "path": ppt_path})

            # 保存類別專屬文件
            category_path = os.path.join(sub_dirs[doc_category], f"{doc_type}_doc_{i+1}.txt")
            with open(category_path, "w", encoding="utf-8") as f:
                f.write(content)

            dataset.append(item)

        # 保存元資料
        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"已生成 {num_items} 個多格式測試文件")
        return dataset
