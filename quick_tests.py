#!/usr/bin/env python3
"""
quick_tests.py

快速測試工具，可選擇執行以下模組：
  - detector  偵測器功能測試
  - replacer  假資料替換測試
  - extractor 文本抽取測試
  - pipeline 端到端 DeID 測試
  - all       一次跑完所有功能

用法範例：
  python quick_tests.py --detector --extractor
  python quick_tests.py --all
"""
import argparse
from pathlib import Path
from src.deid_pipeline.parser.text_extractor import extract_text
from src.deid_pipeline.pii.detectors import get_detector
from src.deid_pipeline.pii.utils.replacer import Replacer
from src.deid_pipeline.image_deid.processor import ImageDeidProcessor
from src.deid_pipeline.config import Config


def test_detector():
    print("\n=== Detector 測試 ===")
    cases = [
        ("zh", "王小明身份證A123456789，手機0912345678", ["ID", "PHONE"]),
        ("en", "Alice lives in Taipei, email alice@mail.com", ["EMAIL", "ADDRESS"]),
    ]
    for lang, txt, expected in cases:
        det = get_detector(lang)
        ents = det.detect(txt)
        types = [e['type'] for e in ents]
        print(f"[{lang}] 原文: {txt}")
        print(f" 偵測到: {types}")
        print(f" 預期: {expected}\n")


def test_replacer():
    print("\n=== Replacer 測試 ===")
    txt = "王小明的電話0912345678"
    det = get_detector("zh").detect(txt)
    fake, events = Replacer().replace(txt, det, mode="replace")
    print(f"原文: {txt}")
    print(f"替換後: {fake}")
    print("事件記錄:")
    for ev in events:
        print(f"  - {ev['original']} => {ev['replacement']} at {ev['span']}")


def test_extractor():
    print("\n=== Text Extractor 測試 ===")
    sample_dir = Path(__file__).parent / 'test_input'
    for file in sample_dir.glob('*'):
        try:
            text, _ = extract_text(file)
            print(f"{file.name}: 提取 {len(text)} 字元")
        except Exception as e:
            print(f"Error on {file.name}: {e}")


def test_image_deid():
    print("\n=== ImageDeID 測試 ===")
    proc = ImageDeidProcessor(lang="zh")
    sample_dir = Path(__file__).parent / 'test_input'
    for img in sample_dir.glob('*.png'):
        print(f"處理影像: {img.name}")
        res = proc.process_image(str(img))
        print(f"  OCR 文字長度: {len(res['original_text'])}")
        print(f"  偵測到 Entities: {[e['type'] for e in res['entities']]}\n")


def main():
    parser = argparse.ArgumentParser(description="Edge DeID Quick Tests")
    parser.add_argument('--detector', action='store_true', help='只跑 PII 偵測測試')
    parser.add_argument('--replacer', action='store_true', help='只跑假資料替換測試')
    parser.add_argument('--extractor', action='store_true', help='只跑文本抽取測試')
    parser.add_argument('--image', action='store_true', help='只跑影像 deid 測試')
    parser.add_argument('--all', action='store_true', help='跑所有測試')
    args = parser.parse_args()

    if args.all or args.detector:
        test_detector()
    if args.all or args.replacer:
        test_replacer()
    if args.all or args.extractor:
        test_extractor()
    if args.all or args.image:
        test_image_deid()

if __name__ == '__main__':
    main()
