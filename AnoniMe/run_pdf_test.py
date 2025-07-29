# run_pdf_detect_test.py
if __name__ == "__main__":
    from file_handlers.pdf_handler import PdfPiiDetector

    # 測試用 PDF 路徑
    input_path = "test_input/sample_test.pdf"
    detector = PdfPiiDetector()
    pii_list = detector.extract_pii(input_path, language="auto")

    # 以易讀格式輸出
    for item in pii_list:
        print(f"Page {item['page']}: {item['entity_type']} ({item['start']}-{item['end']}) -> {item['text']}")

    # 若需存檔，可取消以下註解
    # with open("test_output/sample_test_pii.json", "w", encoding="utf-8") as f:
    #     json.dump(pii_list, f, ensure_ascii=False, indent=2)
    print("\n✅ PII 偵測完成！共偵測到", len(pii_list), "個實體。")
