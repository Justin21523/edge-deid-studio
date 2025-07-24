import argparse
from pathlib import Path

from deid_pipeline.pii import get_detector          # PII 偵測
from deid_pipeline.parser.text_extractor import extract_text
from utils.replacer import Replacer   # 假資料替換


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="輸入檔案路徑")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    ap.add_argument("--mode", choices=["detect", "replace", "black"],
                    default="detect",
                    help="detect=只列實體  replace=偽資料覆寫  black=遮罩範圍")
    ap.add_argument("--json", action="store_true", help="輸出 JSON 事件")
    return ap.parse_args()

def run():
    args = cli()
    fp = Path(args.file)
    if not fp.exists():
        raise FileNotFoundError(fp)

    text = extract_text(str(fp))
    detector = get_detector(args.lang)
    entities = detector.detect(text)

    if args.mode == "detect":
        for ent in entities:
            frag = text[ent["span"][0]:ent["span"][1]]
            print(f"{ent['type']:8s} | {frag} | {ent['score']:.2f}")
    elif args.mode != "detect" and args.json:
        print(Replacer.dumps(events))
    else:
        new_text, events = Replacer().replace(
            text, entities,
            mode="replace" if args.mode == "replace" else "black"
        )
        print("\n===== 置換後文字 =====\n")
        print(new_text)
        print("\n===== 事件列表 =====")
        for ev in events:
            print(ev)

if __name__ == "__main__":
    run()

