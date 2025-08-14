import os
import fitz  # PyMuPDF
from pii_models.presidio_detector import detect_pii
from faker_models.tony_faker import test_all_methods, keep_highest_score_per_raw_txt

class PdfPiiDetector:
    """
    從 PDF 中抽取文字並偵測 PII，回傳可閱讀的列表。
    每個實體包含：頁碼、實體類型、起訖位置、匹配文字。
    """
    def deidentify(self, input_path: str, output_path: str, language: str = "auto") -> str:
        """
        1. 讀取 input_path 的 PDF 檔案
        2. 偵測 PII 並遮蔽
        3. 儲存處理後的 PDF 到 output_path
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到輸入檔：{input_path}")

        doc = fitz.open(input_path)
        new_doc = fitz.open()  # 新 PDF

        for page in doc:
            page_dict = page.get_text("dict")
            new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)
            for block in page_dict["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        print("處理 span：", span)
                        text = span["text"]
                        entities = detect_pii(text, language="en", score_threshold=0.6)
                        print("字串：", text)
                        # print("偵測到的實體：", entities[0].entity_type)
                        # print("偵測到的 PII：", text[entities[0].start:entities[0].end])
                        
                        # 如果沒有偵測到 PII，則直接使用原文字
                        if not entities:
                            masked_text = text
                        else:
                            faker_results = test_all_methods(entities)
                            best_results = keep_highest_score_per_raw_txt(faker_results)
                            # 建立 raw_txt -> fake_value 的 mapping
                            fake_map = {item["raw_txt"]: item["fake_value"] for item in best_results}
                            # 依 entities 位置替換
                            masked_text = text
                            offset = 0
        
                            for ent in entities:
                                start, end = ent["start"] + offset, ent["end"] + offset
                                raw_txt = ent["raw_txt"]
                                entity_type = ent["entity_type"]
                                
                                # 如果是 ORGANIZATION，直接保留原文字
                                if entity_type == "ORGANIZATION":
                                    fake_value = raw_txt
                                else:
                                    fake_value = fake_map.get(raw_txt, "*" * (end - start))
                                    # 確保 fake_value 與 raw_txt 長度一致
                                    fake_value = fake_value[:len(raw_txt)].ljust(len(raw_txt))

                                # 替換並調整 offset
                                masked_text = masked_text[:start] + fake_value + masked_text[end:]
                                offset += len(fake_value) - (end - start)
                            
                        # 用原本的字型、大小、座標插入遮蔽後文字
                        font_path = "/Users/lucasauriant/Downloads/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf"
                        new_page.insert_text(
                            (span["bbox"][0], span["bbox"][1]),
                            masked_text,
                            # fontname=span["font"],  # 使用原檔原字型
                            # 但這裡可能會有問題，因為 PyMuPDF 只支援特定標準字型名稱
                            # 所以這裡改成使用 "helv" or "times" or "cour" 字型
                            # 你可以根據需要改成其他標準字型
                            # or use fontname = "helv"
                            fontname=span["font"],  # 或 "helv", "cour"
                            fontsize=span["size"],
                            fontfile=font_path,
                            color=span.get("color", 0)
                        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_doc.save(output_path)
        return output_path
