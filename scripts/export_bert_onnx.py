#!/usr/bin/env python
"""
Export ckiplab/bert-base-chinese-ner to ONNX (opset17).
Then quantize to INT8 weights.
"""
import subprocess
import sys

def run(cmd):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)

if __name__ == "__main__":
    # 1. Export to ONNX
    run(
        "optimum-cli export onnx "
        "--model ckiplab/bert-base-chinese-ner "
        "--task token-classification "
        "--opset 17 "
        "models/bert_ner_zh"
    )
    # 2. Quantize to INT8
    run(
        "optimum-cli onnxruntime quantize "
        "--model models/bert_ner_zh "
        "--weight-only "
        "--per-channel "
        "--dtype qint8 "
        "--output models/bert_ner_zh_q"
    )
    print("✅ export + quantize done")
