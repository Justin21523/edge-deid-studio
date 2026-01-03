from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from .base import ExtractedDocument, SegmentSpec, build_document_from_segments


class ImageHandler:
    extensions = [".png", ".jpg", ".jpeg", ".bmp"]

    def extract(self, input_path: Path, *, language: str) -> ExtractedDocument:
        from ..parser.ocr import get_ocr_reader

        reader = get_ocr_reader(engine="auto", lang=language)
        full_text, blocks = reader.recognize(str(input_path))

        segments: list[SegmentSpec] = []
        for block in blocks:
            block_text = str(block.get("text", "")).strip()
            if not block_text:
                continue
            left = int(block["left"])
            top = int(block["top"])
            right = int(block["left"] + block["width"])
            bottom = int(block["top"] + block["height"])
            segments.append(
                SegmentSpec(
                    text=block_text,
                    page_index=0,
                    bbox=(left, top, right, bottom),
                    metadata={"ocr_conf": float(block.get("conf", 0.0))},
                )
            )

        return build_document_from_segments(
            input_path=input_path,
            language=language,
            segments=segments or [SegmentSpec(text=full_text or "")],
            separator=" ",
        )

    def rebuild(
        self,
        document: ExtractedDocument,
        *,
        output_text: str,
        entities: list[dict],
        replacement_map: Dict[str, str],
        events: list[dict],
        output_dir: Optional[Path] = None,
        mode: str = "replace",
    ) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {"output_text": output_text}

        img = cv2.imread(str(document.input_path))
        if img is None:
            return artifacts

        redaction_rects: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int, int]] = set()

        for entity in entities:
            entity_type = str(entity.get("type", "UNKNOWN"))
            bbox = entity.get("bbox")
            if bbox:
                try:
                    left, top, right, bottom = map(int, bbox)
                except Exception:
                    continue
                rect_key = (left, top, right, bottom)
                if rect_key not in seen:
                    seen.add(rect_key)
                    redaction_rects.append(
                        {
                            "page_index": int(entity.get("page_index", 0)),
                            "bbox": rect_key,
                            "entity_type": entity_type,
                            "strategy": "entity_bbox",
                        }
                    )
                continue

            span = entity.get("span")
            if not span:
                continue

            try:
                start, end = map(int, span)
            except Exception:
                continue

            entity_page = entity.get("page_index")
            for segment in document.segments:
                if segment.bbox is None:
                    continue
                if entity_page is not None and segment.page_index is not None:
                    if int(entity_page) != int(segment.page_index):
                        continue
                if segment.end <= start or segment.start >= end:
                    continue

                left, top, right, bottom = map(int, segment.bbox)
                rect_key = (left, top, right, bottom)
                if rect_key in seen:
                    continue
                seen.add(rect_key)
                redaction_rects.append(
                    {
                        "page_index": int(segment.page_index or 0),
                        "bbox": rect_key,
                        "entity_type": entity_type,
                        "strategy": "segment_overlap",
                        "segment_id": int(segment.segment_id),
                    }
                )

        for rect in redaction_rects:
            left, top, right, bottom = rect["bbox"]
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 0), thickness=-1)

        ok, buffer = cv2.imencode(".jpg", img)
        if ok:
            artifacts["visual_result"] = buffer.tobytes()
            artifacts["visual_format"] = "jpg"

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{document.input_path.stem}.deid.jpg"
            cv2.imwrite(str(out_path), img)
            artifacts["output_path"] = str(out_path)
            artifacts["rebuild_supported"] = True
        else:
            artifacts["rebuild_supported"] = False

        artifacts["rebuild_mode"] = "redact_bbox"
        artifacts["redaction_count"] = len(redaction_rects)
        artifacts["redaction_rects"] = redaction_rects
        return artifacts
