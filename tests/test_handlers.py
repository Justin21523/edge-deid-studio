from __future__ import annotations

from pathlib import Path

import pytest

from deid_pipeline.handlers.registry import DEFAULT_REGISTRY
from deid_pipeline.handlers.text import TextHandler


def test_registry_selects_text_handler():
    handler = DEFAULT_REGISTRY.get(Path("sample.txt"))
    assert isinstance(handler, TextHandler)


def test_registry_unknown_extension_raises():
    with pytest.raises(ValueError):
        DEFAULT_REGISTRY.get(Path("sample.unknown"))


