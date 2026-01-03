from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Type

from .base import FormatHandler
from .csv import CsvHandler
from .docx import DocxHandler
from .image import ImageHandler
from .pdf import PdfHandler
from .pptx import PptxHandler
from .text import TextHandler
from .xlsx import XlsxHandler


class HandlerRegistry:
    def __init__(self) -> None:
        self._by_ext: Dict[str, Type[FormatHandler]] = {}

    def register(self, handler_cls: Type[FormatHandler]) -> None:
        for ext in handler_cls.extensions:
            self._by_ext[ext.lower()] = handler_cls

    def get(self, input_path: Path) -> FormatHandler:
        ext = input_path.suffix.lower()
        handler_cls = self._by_ext.get(ext)
        if handler_cls is None:
            raise ValueError(f"No handler registered for extension: {ext}")
        return handler_cls()


def default_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    for handler_cls in [
        TextHandler,
        CsvHandler,
        PdfHandler,
        ImageHandler,
        DocxHandler,
        XlsxHandler,
        PptxHandler,
    ]:
        registry.register(handler_cls)
    return registry


DEFAULT_REGISTRY = default_registry()
