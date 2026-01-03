#!/usr/bin/env python3
"""
quick_tests.py

Quick smoke test runner for EdgeDeID Studio.

Available suites:
  - detector  Detector sanity checks
  - replacer  Replacement sanity checks
  - extractor Extraction sanity checks (handlers)
  - pipeline  End-to-end DeidPipeline smoke tests
  - image     Image pipeline smoke tests
  - all       Run everything

Examples:
  python quick_tests.py --detector --extractor
  python quick_tests.py --all
"""
import argparse
from pathlib import Path

from deid_pipeline import DeidPipeline
from deid_pipeline.pii.detectors import get_detector
from deid_pipeline.pii.utils.replacer import Replacer
from deid_pipeline.image_deid.processor import ImageDeidProcessor


def test_detector():
    print("\n=== Detector Test ===")
    cases = [
        ("zh", "\u738b\u5c0f\u660e\u8eab\u4efd\u8b49A123456789\uff0c\u624b\u6a5f0912345678", ["ID", "PHONE"]),
        ("en", "Alice lives in Taipei, email alice@mail.com", ["EMAIL", "ADDRESS"]),
    ]
    for lang, txt, expected in cases:
        det = get_detector(lang)
        ents = det.detect(txt)
        types = [e["type"] for e in ents]
        print(f"[{lang}] Text: {txt}")
        print(f" Detected: {types}")
        print(f" Expected: {expected}\n")


def test_replacer():
    print("\n=== Replacer Test ===")
    txt = "\u738b\u5c0f\u660e\u7684\u96fb\u8a710912345678"
    det = get_detector("zh").detect(txt)
    fake, events = Replacer().replace(txt, det, mode="replace")
    print(f"Original: {txt}")
    print(f"Replaced: {fake}")
    print("Events:")
    for ev in events:
        print(f"  - {ev.get('original')} => {ev.get('replacement')} at {ev.get('span')}")


def test_extractor():
    print("\n=== Extraction Test (Handlers) ===")
    sample_dir = Path(__file__).parent / "test_input"
    pipeline = DeidPipeline(language="zh")
    registry = pipeline._get_handler_registry()

    for file in sorted(sample_dir.glob("*")):
        try:
            handler = registry.get(file)
            doc = handler.extract(file, language="zh")
            print(f"{file.name}: extracted {len(doc.text)} chars ({doc.file_extension})")
        except Exception as e:
            print(f"{file.name}: error: {e}")


def test_image_deid():
    print("\n=== Image DeID Test ===")
    proc = ImageDeidProcessor(lang="zh")
    sample_dir = Path(__file__).parent / "test_input"
    for img in sorted(sample_dir.glob("*.png")):
        print(f"Processing image: {img.name}")
        res = proc.process_image(str(img))
        print(f"  OCR text length: {len(res.get('original_text',''))}")
        print(f"  Entities: {[e['type'] for e in res.get('entities', [])]}\n")


def test_pipeline():
    print("\n=== Pipeline Test ===")
    sample_dir = Path(__file__).parent / "test_input"
    pipeline = DeidPipeline(language="zh")

    for file in sorted(sample_dir.glob("*")):
        if file.suffix.lower() not in {".txt", ".csv", ".docx", ".pdf", ".png", ".jpg", ".jpeg", ".xlsx", ".pptx"}:
            continue
        try:
            result = pipeline.process(str(file), output_mode="replace", output_dir=None)
            print(f"{file.name}: ok | text_len={len(result.text)} | entities={len(result.entities)}")
        except Exception as exc:
            print(f"{file.name}: error: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Edge DeID Quick Tests")
    parser.add_argument("--detector", action="store_true", help="Run detector smoke tests only")
    parser.add_argument("--replacer", action="store_true", help="Run replacement smoke tests only")
    parser.add_argument("--extractor", action="store_true", help="Run extraction smoke tests only")
    parser.add_argument("--pipeline", action="store_true", help="Run end-to-end pipeline smoke tests only")
    parser.add_argument("--image", action="store_true", help="Run image smoke tests only")
    parser.add_argument("--all", action="store_true", help="Run all smoke tests")
    args = parser.parse_args()

    if args.all or args.detector:
        test_detector()
    if args.all or args.replacer:
        test_replacer()
    if args.all or args.extractor:
        test_extractor()
    if args.all or args.pipeline:
        test_pipeline()
    if args.all or args.image:
        test_image_deid()


if __name__ == "__main__":
    main()
