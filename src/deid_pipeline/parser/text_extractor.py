import os
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from docx import Document
import fitz  # PyMuPDF
from .ocr import get_ocr_reader
from ..config import OCR_THRESHOLD, USE_STUB

# only text will be extracted!
# 圖片裡面的文字無法提取
def extract_text(file_path: str, ocr: bool=False) -> str:
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    # pdf 如果太過於複雜(格子過多、編排多樣)，可能讀取到的資料就會沒有按照邏輯
    # 這一行是「身分證字號」，下一行不會是「數字」，可能是別的欄位或別的文字內容
    elif ext == ".pdf":
        try:
            pdf = fitz.open(file_path)
        except fitz.FileNotFoundError:
            raise RuntimeError(f"File not found: {file_path}")
        except fitz.FileDataError:
            raise RuntimeError(f"Corrupted PDF: {file_path}")

        full_text = []
        for page in pdf:
            # 1) 先做「區塊排序」提取
            blocks = page.get_text("blocks", sort=True)
            page_text = "\n".join(b[4] for b in blocks if b[4].strip())

            # 2) 少量文字才觸發 OCR
            if ocr and not USE_STUB and len(page_text) < OCR_THRESHOLD:
                reader = get_ocr_reader()
                img = page.get_pixmap().samples
                h, w = int(page.rect.height), int(page.rect.width)
                arr = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 3))
                ocr_lines = [t[1] for t in reader.readtext(arr)]
                full_text.append("\n".join(ocr_lines))
            else:
                full_text.append(page_text)

        return "\n".join(full_text)

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_string(index=False)

    # 只有 cover 到 excel 裡面的第一個工作表
    # 其餘的工作表內容不會被讀取到
    elif ext == ".xlsx":
        df = pd.read_excel(file_path)
        return df.to_string(index=False)

    # 如果是 HTML 裡面有圖片的話， <img src> 裡面的 alt 也就是圖片的文字無法顯示的時候會跳出的替代文字，會讀取不到。
    elif ext in [".html", ".xml"]:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        return soup.get_text()

    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def save_as_txt(text: str, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


# 🚀 測試執行：只要提供 input_path
if __name__ == "__main__":
    input_path = input("輸入檔案路徑：").strip()
    output_path = os.path.splitext(input_path)[0] + "_extracted.txt"

    try:
        content = extract_text(input_path)
        save_as_txt(content, output_path)
        print(f"✅ 成功儲存純文字到：{output_path}")
    except Exception as e:
        print(f"❌ 錯誤：{e}")
