# file_handlers/docx_handler.py

import os
from docx import Document
from pii_models.presidio_detector import detect_pii
from faker_models.presidio_replacer import replace_pii

class DocxHandler:
    """
    處理 .docx 檔案 in-place 去識別化，
    保留所有段落格式、run 樣式與表格結構。
    """

    def deidentify(self, input_path: str, output_path: str, language: str = "auto") -> str:
        """
        1. 讀取 input_path 的 Word 文件
        2. 針對每個 run.text 呼叫 detect_pii() + replace_pii()
        3. 處理表格內 cell
        4. 存成 output_path，並回傳它
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到輸入檔：{input_path}")

        doc = Document(input_path)

        # 處理段落中的 runs
        for para in doc.paragraphs:
            for run in para.runs:
                original = run.text
                entities = detect_pii(original, language="en", score_threshold=0.6)
                # print("處理段落 run：", original)
                # print("偵測到的實體：", entities)
                new_text = replace_pii(original, entities)
                if new_text != original:
                    run.text = new_text

        # 處理表格
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            original = run.text
                            entities = detect_pii(original, language="en", score_threshold=0.6)
                            # print("處理表格 cell run：", original)
                            # print("偵測到的實體：", entities)
                            new_text = replace_pii(original, entities)
                            if new_text != original:
                                run.text = new_text

        # 儲存結果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc.save(output_path)
        return output_path
