# run_docx_test.py

import os
from file_handlers.docx_handler import DocxHandler

def main():
    # 1) 指定要處理的 Word 檔案
    input_path = "test_input/pii_test.docx"
    # 2) 指定輸出路徑
    output_path = "test_output/pii_test_deid.docx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3) 建立 handler 並執行去識別化
    handler = DocxHandler()
    handler.deidentify(input_path=input_path, output_path=output_path)

    print("✅ 去識別化完成！輸出檔案：", output_path)

if __name__ == "__main__":
    main()
