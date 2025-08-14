# run_txt_test.py
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from file_handlers.txt_handler import TextHandler

if __name__ == "__main__":
    # 測試示例
    input_path = "test_input/test_extracted.txt"
    output_path = "test_output/sample_deid.txt"
    handler = TextHandler()
    handler.deidentify(input_path=input_path, output_path=output_path)
    print(f"✅ 去識別化完成，輸出檔: {output_path}")
