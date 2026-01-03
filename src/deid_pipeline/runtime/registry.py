from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple

from .onnx import create_session_options, select_onnx_providers


@lru_cache(maxsize=8)
def get_hf_tokenizer(model_dir: str):
    """Return a cached Hugging Face tokenizer (local-only)."""

    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)


@lru_cache(maxsize=4)
def get_hf_token_classifier(model_dir: str):
    """Return a cached Hugging Face token classification model (local-only)."""

    from transformers import AutoModelForTokenClassification  # type: ignore

    return AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)


@lru_cache(maxsize=2)
def get_spacy_pipeline(model_name: str):
    """Return a cached spaCy pipeline."""

    import spacy  # type: ignore

    return spacy.load(model_name)


@lru_cache(maxsize=4)
def get_onnx_session(
    onnx_model_path: str,
    providers: Tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
):
    """Return a cached ONNX Runtime inference session."""

    import onnxruntime as ort  # type: ignore

    resolved_providers = select_onnx_providers(providers)
    sess_options = create_session_options()

    return ort.InferenceSession(
        onnx_model_path, sess_options=sess_options, providers=list(resolved_providers)
    )


def ensure_local_path(path: Path) -> str:
    """Resolve a local directory path string for Hugging Face loaders."""

    resolved = path.expanduser().resolve()
    return str(resolved)
