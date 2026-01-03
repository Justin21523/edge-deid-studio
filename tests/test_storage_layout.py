from __future__ import annotations

import os
from pathlib import Path

from deid_pipeline.storage.env import apply_cache_env_defaults, cache_env_defaults
from deid_pipeline.storage.layout import StorageLayout, dataset_slug


def test_dataset_slug_normalizes_identifiers():
    assert dataset_slug("tner/wikiann") == "tner__wikiann"
    assert dataset_slug("a:b c") == "a__b_c"
    assert dataset_slug("") == "unknown"


def test_storage_layout_respects_env_overrides(tmp_path: Path, monkeypatch):
    cache_home = tmp_path / "cache"
    models_home = tmp_path / "models"
    data_home = tmp_path / "data"
    cache_home.mkdir()
    models_home.mkdir()
    data_home.mkdir()

    monkeypatch.setenv("EDGE_DEID_CACHE_HOME", str(cache_home))
    monkeypatch.setenv("EDGE_DEID_MODELS_HOME", str(models_home))
    monkeypatch.setenv("EDGE_DEID_DATA_HOME", str(data_home))

    layout = StorageLayout.from_project_root(tmp_path)
    assert layout.cache_home == cache_home.resolve()
    assert layout.models_home == models_home.resolve()
    assert layout.data_home == data_home.resolve()


def test_storage_layout_dataset_and_training_overrides(tmp_path: Path, monkeypatch):
    layout = StorageLayout(
        project_root=tmp_path,
        cache_home=tmp_path / "cache",
        models_home=tmp_path / "models",
        data_home=tmp_path / "data",
    )

    monkeypatch.setenv("EDGE_DEID_DATASETS_HOME", str(tmp_path / "custom_datasets"))
    monkeypatch.setenv("EDGE_DEID_TRAINING_HOME", str(tmp_path / "custom_training"))

    assert layout.datasets_home == (tmp_path / "custom_datasets").resolve()
    assert layout.training_home == (tmp_path / "custom_training").resolve()
    assert layout.training_runs_home == (tmp_path / "custom_training" / "runs").resolve()
    assert layout.training_logs_home == (tmp_path / "custom_training" / "logs").resolve()


def test_cache_env_defaults_does_not_mutate_environment(tmp_path: Path, monkeypatch):
    layout = StorageLayout(
        project_root=tmp_path,
        cache_home=tmp_path / "cache",
        models_home=tmp_path / "models",
        data_home=tmp_path / "data",
    )

    for key in ["HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "PIP_CACHE_DIR"]:
        monkeypatch.delenv(key, raising=False)

    values = cache_env_defaults(layout=layout)
    assert os.getenv("HF_HOME") is None
    assert values["HF_HOME"].endswith("/cache/huggingface")


def test_apply_cache_env_defaults_sets_vars_and_creates_dirs(tmp_path: Path, monkeypatch):
    layout = StorageLayout(
        project_root=tmp_path,
        cache_home=tmp_path / "cache",
        models_home=tmp_path / "models",
        data_home=tmp_path / "data",
    )

    for key in ["HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "PIP_CACHE_DIR"]:
        monkeypatch.delenv(key, raising=False)

    applied = apply_cache_env_defaults(layout=layout, overwrite=False, create_dirs=True)

    for key, value in applied.items():
        assert os.environ[key] == value

    assert Path(os.environ["HF_HOME"]).exists()
    assert Path(os.environ["TORCH_HOME"]).exists()
    assert Path(os.environ["XDG_CACHE_HOME"]).exists()
    assert Path(os.environ["PIP_CACHE_DIR"]).exists()

