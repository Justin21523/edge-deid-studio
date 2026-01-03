from __future__ import annotations

import os
import sys
from pathlib import Path


def pytest_configure() -> None:
    """Test bootstrap.

    - Ensure the `src/` layout is importable without requiring `pip install -e .`
    - Default to offline mode to prevent accidental network calls during tests
    """

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Keep the default test mode lightweight.
    os.environ.setdefault("USE_STUB", "true")
    os.environ.setdefault("USE_ONNX", "false")
    os.environ.setdefault("USE_SPACY", "false")
    os.environ.setdefault("OCR_ENABLED", "false")

