from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExtractedDocument, SegmentSpec, build_document_from_segments


class TextHandler:
    extensions = [".txt", ".html"]

    def extract(self, input_path: Path, *, language: str) -> ExtractedDocument:
        ext = input_path.suffix.lower()
        if ext == ".txt":
            text = input_path.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                from bs4 import BeautifulSoup  # type: ignore
            except Exception as exc:
                raise ImportError("beautifulsoup4 is required to extract HTML files") from exc

            html = input_path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")

        return build_document_from_segments(
            input_path=input_path,
            language=language,
            segments=[SegmentSpec(text=text)],
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
        # Phase 2 baseline: return text artifacts only.
        artifacts: Dict[str, Any] = {
            "output_text": output_text,
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{document.input_path.stem}.deid{document.file_extension}"
            out_path.write_text(output_text, encoding="utf-8")
            artifacts["output_path"] = str(out_path)
        return artifacts
