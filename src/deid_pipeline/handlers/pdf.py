from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExtractedDocument, SegmentSpec, build_document_from_segments


class PdfHandler:
    extensions = [".pdf"]

    def extract(self, input_path: Path, *, language: str) -> ExtractedDocument:
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise ImportError("PyMuPDF (fitz) is required to extract PDF files") from exc

        doc = fitz.open(str(input_path))
        segments: list[SegmentSpec] = []

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            blocks = page.get_text("blocks", sort=True)
            for block in blocks:
                if len(block) < 7:
                    continue
                if block[6] != 0:
                    continue
                block_text = str(block[4]).strip()
                if not block_text:
                    continue
                bbox = (int(block[0]), int(block[1]), int(block[2]), int(block[3]))
                segments.append(SegmentSpec(text=block_text, page_index=page_index, bbox=bbox))

        return build_document_from_segments(
            input_path=input_path,
            language=language,
            segments=segments or [SegmentSpec(text="")],
            separator="\n",
            metadata={"pages": len(doc)},
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
        try:
            import fitz  # type: ignore
        except Exception:
            return {"output_text": output_text, "rebuild_supported": False}

        if output_dir is None:
            return {"output_text": output_text, "rebuild_supported": False}

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{document.input_path.stem}.deid.pdf"

        src_doc = fitz.open(str(document.input_path))

        # Precision improvement: attempt a text search to narrow redaction rectangles.
        # If search fails, fall back to the segment-level bounding box anchor.
        redaction_rects: list[dict[str, Any]] = []
        rects_by_page: dict[int, list[Any]] = defaultdict(list)

        entities_by_page: dict[int, list[dict]] = defaultdict(list)
        for ent in entities:
            page_index = ent.get("page_index")
            if page_index is None:
                continue
            try:
                entities_by_page[int(page_index)].append(ent)
            except Exception:
                continue

        for page_index, page_entities in entities_by_page.items():
            try:
                page = src_doc.load_page(int(page_index))
            except Exception:
                continue

            for ent in page_entities:
                bbox = ent.get("bbox")
                if bbox is None:
                    continue

                try:
                    block_rect = fitz.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                except Exception:
                    continue

                needle = str(ent.get("text") or "").strip()
                candidate_rects: list[Any] = []

                if needle:
                    try:
                        found_rects = page.search_for(needle)
                    except Exception:
                        found_rects = []

                    candidate_rects = [r for r in found_rects if r.intersects(block_rect)]

                    if not candidate_rects:
                        normalized_needle = " ".join(needle.split())
                        if normalized_needle and normalized_needle != needle:
                            try:
                                found_rects = page.search_for(normalized_needle)
                            except Exception:
                                found_rects = []
                            candidate_rects = [r for r in found_rects if r.intersects(block_rect)]

                if not candidate_rects:
                    candidate_rects = [block_rect]

                for rect in candidate_rects:
                    rects_by_page[page_index].append(rect)
                    redaction_rects.append(
                        {
                            "page_index": int(page_index),
                            "bbox": (int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1)),
                            "strategy": "search_for" if needle and rect is not block_rect else "block_bbox",
                        }
                    )

        for page_index, rects in rects_by_page.items():
            try:
                page = src_doc.load_page(int(page_index))
            except Exception:
                continue

            for rect in rects:
                try:
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                except Exception:
                    continue

            try:
                page.apply_redactions()
            except Exception:
                continue

        src_doc.save(str(out_path))
        src_doc.close()

        return {
            "output_text": output_text,
            "output_path": str(out_path),
            "rebuild_supported": True,
            "rebuild_mode": "redact",
            "redaction_count": len(redaction_rects),
            "redaction_rects": redaction_rects,
        }
