# pii_models/detector.py

import spacy
import re

# 載入英文模型（可換成 zh_core_web_trf 若處理繁中）
nlp = spacy.load("en_core_web_sm")

# 可加入正則規則（電話、email、身分證、信用卡等）
PII_PATTERNS = {
    "PHONE": [
        r"\b09\d{8}\b",
        r"\b\d{3}-\d{3}-\d{4}\b",
        r"\+9\d{8}\b"
    ],
    "EMAIL": [
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    ],
    "ID": [
        r"\b[A-Z][1-2]\d{8}\b"
    ]
}


def detect_pii(text):
    doc = nlp(text)
    results = []

    # 使用 spaCy NER
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG", "LOC"]:
            results.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "text": ent.text
            })

    # 加上自訂規則
    # 修改後的迴圈：
    for label, patterns in PII_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                results.append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": label,
                    "text": match.group()
                })

    # 排序（保證後續替代時不出錯）
    results = sorted(results, key=lambda x: x["start"])
    return results
