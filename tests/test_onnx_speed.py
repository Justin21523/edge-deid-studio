import time
import numpy as np
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

def test_latency():
    model_dir = "models/bert_ner_zh_q"
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = ORTModelForTokenClassification.from_pretrained(model_dir)

    sentence = "王小明的身分證號是A123456789。"
    inputs = tok(sentence, return_tensors="pt")

    # 熱機三次
    for _ in range(3):
        _ = model(**inputs)

    t0 = time.time()
    for _ in range(100):
        _ = model(**inputs)
    t1 = time.time()
    avg_ms = (t1 - t0) / 100 * 1000
    print(f"平均延遲(ms): {avg_ms:.2f}")

    # 驗證 CPU 延遲不超過 25ms
    assert avg_ms < 25, f"延遲過高：{avg_ms:.2f} ms"
