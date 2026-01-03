from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExtractedDocument, SegmentSpec, build_document_from_segments
from ..core.contracts import replacement_key


class CsvHandler:
    extensions = [".csv"]

    def extract(self, input_path: Path, *, language: str) -> ExtractedDocument:
        segments: list[SegmentSpec] = []
        with input_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader, start=1):
                for col_idx, value in enumerate(row, start=1):
                    text = str(value or "")
                    segments.append(
                        SegmentSpec(
                            text=text,
                            cell={
                                "sheet": "csv",
                                "row": row_idx,
                                "col": col_idx,
                                "address": f"R{row_idx}C{col_idx}",
                            },
                        )
                    )

        return build_document_from_segments(
            input_path=input_path,
            language=language,
            segments=segments or [SegmentSpec(text="")],
            separator="\n",
            metadata={"format": "csv"},
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

        if output_dir is None:
            artifacts["rebuild_supported"] = False
            return artifacts

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{document.input_path.stem}.deid.csv"

        with document.input_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))

        for entity in entities:
            cell = entity.get("cell")
            if not cell:
                continue
            row = int(cell.get("row", 0))
            col = int(cell.get("col", 0))
            if row <= 0 or col <= 0:
                continue
            if row > len(rows) or col > len(rows[row - 1]):
                continue

            original = entity.get("text")
            entity_type = entity.get("type")
            if not original or not entity_type:
                continue

            replacement = replacement_map.get(replacement_key(str(entity_type), str(original)))
            if replacement is None:
                continue

            current_value = str(rows[row - 1][col - 1] or "")
            rows[row - 1][col - 1] = current_value.replace(str(original), str(replacement), 1)

        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        artifacts["output_path"] = str(out_path)
        artifacts["rebuild_supported"] = True
        return artifacts
