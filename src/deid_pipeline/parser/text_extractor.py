import os
import json
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
import fitz  # PyMuPDF

# only text will be extracted!
# 圖片裡面的文字無法提取
def extract_text(file_path: str) -> str:
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
        pdf = fitz.open(file_path)
        return "\n".join([page.get_text() for page in pdf])

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

