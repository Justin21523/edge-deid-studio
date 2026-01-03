from __future__ import annotations

from pathlib import Path

from deid_pipeline.training.artifacts import copy_support_files


def test_copy_support_files_excludes_weights(tmp_path: Path):
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()

    # Support files
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "vocab.txt").write_text("hello\n", encoding="utf-8")
    (model_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")

    # Weight-ish files that should not be copied
    (model_dir / "pytorch_model.bin").write_bytes(b"\x00\x01")
    (model_dir / "model.safetensors").write_bytes(b"\x00\x01")

    copied = copy_support_files(model_dir, out_dir, overwrite=True)
    copied_names = {p.name for p in copied}

    assert "config.json" in copied_names
    assert "tokenizer.json" in copied_names
    assert "vocab.txt" in copied_names
    assert "merges.txt" in copied_names

    assert not (out_dir / "pytorch_model.bin").exists()
    assert not (out_dir / "model.safetensors").exists()

