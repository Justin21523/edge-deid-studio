# run_pdf_detect_test.py
import os, sys
# 把项目根（脚本的上一层）加入 module 搜索路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from file_handlers.pdf_handler_1 import PdfPiiDetector


def main():

    # 測試用 PDF input 路徑
    input_path = "/Users/lucasauriant/Desktop/AnoniMe/test_input/sample_test.pdf"
    # 測試 output 路徑
    output_path = "test_output/sample_test_deid.pdf"
    
    detector = PdfPiiDetector()
    
    # 執行去識別化
    detector.deidentify(input_path=input_path, output_path=output_path)
    
    # pii_list = detector.extract_pii(input_path=input_path, output_path=output_path)
    # 以易讀格式輸出
    # for item in pii_list:
    #     print(f"Page {item['page']}: {item['entity_type']} ({item['start']}-{item['end']}) -> {item['text']}")

    # 若需存檔，可取消以下註解
    # with open("test_output/sample_test_pii.json", "w", encoding="utf-8") as f:
    #     json.dump(pii_list, f, ensure_ascii=False, indent=2)
    # print("\n✅ PII 偵測完成！共偵測到", len(pii_list), "個實體。")
    print("✅ 去識別化完成！輸出檔案：", output_path)

if __name__ == "__main__":
    main()
