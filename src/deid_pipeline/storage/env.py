from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

from .layout import StorageLayout


def cache_env_defaults(*, layout: StorageLayout) -> Dict[str, str]:
    """Return recommended cache env vars (without mutating the process environment)."""

    cache_home = layout.cache_home
    hf_home = cache_home / "huggingface"
    torch_home = cache_home / "torch"
    pip_cache = cache_home / "pip"

    return {
        "HF_HOME": str(hf_home),
        "TRANSFORMERS_CACHE": str(hf_home),
        "TORCH_HOME": str(torch_home),
        "XDG_CACHE_HOME": str(cache_home),
        "PIP_CACHE_DIR": str(pip_cache),
    }


def apply_cache_env_defaults(
    *,
    layout: StorageLayout,
    overwrite: bool = False,
    create_dirs: bool = True,
) -> Mapping[str, str]:
    """Apply cache env defaults for dev tooling to avoid writing under $HOME/.cache.

    This should be called by scripts/notebooks before importing HF/torch/onnx stacks.
    """

    values = cache_env_defaults(layout=layout)
    for key, value in values.items():
        if overwrite:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)

    if create_dirs:
        for key in ["HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "PIP_CACHE_DIR"]:
            Path(os.environ[key]).expanduser().mkdir(parents=True, exist_ok=True)

    return values

