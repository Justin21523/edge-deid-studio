# main.py
import argparse
from pathlib import Path

from deid_pipeline.parser.text_extractor import extract_text
from deid_pipeline.pii import get_detector
from deid_pipeline.pii.utils.replacer import Replacer

def parse_args():
    p = argparse.ArgumentParser(description="De-ID CLI")
    p.add_argument("-i", "--input", required=True, help="Input file path (txt|docx|pdf|png|jpg)")
    p.add_argument("-l", "--lang", choices=["zh","en"], default="zh", help="Language for detection")
    p.add_argument("-m", "--mode", choices=["detect","replace","black"], default="replace",
                   help="detect: list PII, replace: substitute, black: output mask spans")
    p.add_argument("--json", action="store_true", help="When replace/black, also print JSON events")
    return p.parse_args()

def main():
    args = parse_args()
    fp = Path(args.input)
    if not fp.exists():
        raise FileNotFoundError(f"Input not found: {fp}")

    # 1. Extract text (for PDF/TXT/DOCX) or OCR (for images)
    text, offsets = extract_text(str(fp))

    # 2. Detect PII
    detector = get_detector(args.lang)
    entities = detector.detect(text)

    # 3. Replace or mask
    replacer = Replacer()
    if args.mode == "detect":
        # just list out
        for ent in entities:
            s,e = ent["span"]
            snippet = text[s:e]
            print(f"{ent['type']:10} | {snippet} | {ent['score']:.2f}")
    else:
        new_text, events = replacer.replace(
            text,
            entities,
            mode="replace" if args.mode=="replace" else "black"
        )
        print("\n===== Result Text =====\n")
        print(new_text)
        if args.json:
            print("\n===== Events JSON =====\n")
            print(Replacer.dumps(events))

if __name__ == "__main__":
    main()
