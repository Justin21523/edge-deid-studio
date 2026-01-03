from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List, Set


MODEL_WEIGHT_SUFFIXES: Set[str] = {".bin", ".pt", ".pth", ".ckpt", ".safetensors"}

SUPPORT_FILE_NAMES: Set[str] = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.txt",
    "vocab.json",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
}


def should_copy_support_file(path: Path) -> bool:
    """Return True when a file should be copied into a packaged runtime artifact."""

    if not path.is_file():
        return False

    if path.suffix.lower() in MODEL_WEIGHT_SUFFIXES:
        return False

    name = path.name
    if name in SUPPORT_FILE_NAMES:
        return True

    if name.startswith("tokenizer"):
        return True

    if name.startswith("vocab"):
        return True

    if name.endswith(".model"):
        return True

    return False


def copy_support_files(
    model_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = True,
) -> List[Path]:
    """Copy tokenizer/config support files from a model directory into an output directory."""

    src = Path(model_dir).expanduser().resolve()
    dst = Path(output_dir).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    for item in src.iterdir():
        if not should_copy_support_file(item):
            continue

        target = dst / item.name
        if target.exists() and not overwrite:
            continue

        shutil.copy2(str(item), str(target))
        copied.append(target)

    return copied

