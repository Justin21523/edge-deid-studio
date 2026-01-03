from __future__ import annotations

import argparse
import json
from pathlib import Path

from deid_pipeline import DeidPipeline
from deid_pipeline.core.anchors import attach_segment_anchors
from deid_pipeline.core.contracts import normalize_entity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EdgeDeID CLI (offline-first)")
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input file path (txt|html|csv|pdf|png|jpg|docx|xlsx|pptx)",
    )
    p.add_argument(
        "-l",
        "--lang",
        choices=["zh", "en"],
        default="zh",
        help="Language for detection (default: zh)",
    )
    p.add_argument(
        "-m",
        "--mode",
        choices=["detect", "replace", "black"],
        default="replace",
        help="detect: list entities; replace: substitute; black: mask spans",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for rebuilt artifacts (e.g., redacted PDF, rewritten CSV).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output (entities for detect; events for replace/black).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fp = Path(args.input)
    if not fp.exists():
        raise FileNotFoundError(f"Input not found: {fp}")

    pipeline = DeidPipeline(language=args.lang)

    if args.mode == "detect":
        registry = pipeline._get_handler_registry()
        handler = registry.get(fp)
        document = handler.extract(fp, language=args.lang)
        text = document.text
        detector = pipeline._get_detector()
        raw_entities = detector.detect(text)
        entities = [normalize_entity(e, language=args.lang, text=text) for e in raw_entities]
        attach_segment_anchors(entities, document.segments)

        for ent in entities:
            s, e = ent.get("span", (0, 0))
            snippet = text[s:e]
            conf = float(ent.get("confidence", ent.get("score", 0.0)))
            print(f"{ent.get('type','UNKNOWN'):16} | {snippet} | {conf:.2f}")

        if args.json:
            print(json.dumps(entities, ensure_ascii=False, indent=2))
        return

    output_mode = "replace" if args.mode == "replace" else "blackbox"
    result = pipeline.process(
        str(fp),
        output_mode=output_mode,
        output_dir=args.output_dir,
    )

    print("\n===== Result Text =====\n")
    print(result.text)

    if args.json:
        print("\n===== Events JSON =====\n")
        print(json.dumps(result.events, ensure_ascii=False, indent=2))
        if result.artifacts:
            print("\n===== Artifacts JSON =====\n")
            print(json.dumps(result.artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
