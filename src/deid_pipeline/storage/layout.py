from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _first_existing_dir(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and path.is_dir():
            return path.resolve()
    return None


def dataset_slug(name: str) -> str:
    """Return a filesystem-safe dataset slug from a dataset identifier."""

    cleaned = (name or "").strip()
    if not cleaned:
        return "unknown"
    return cleaned.replace("/", "__").replace(":", "__").replace(" ", "_")


@dataclass(frozen=True)
class StorageLayout:
    """Resolve dataset/model/cache roots based on the AI_WAREHOUSE 3.0 layout.

    Defaults follow the spec in `~/Desktop/data_model_structure.md`:
    - caches/models/code: `/mnt/c`
    - datasets/training outputs: `/mnt/data`
    """

    project_root: Path
    cache_home: Path
    models_home: Path
    data_home: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "StorageLayout":
        root = Path(project_root).expanduser().resolve()

        cache_home = (
            _first_existing_dir(os.getenv("EDGE_DEID_CACHE_HOME"), os.getenv("XDG_CACHE_HOME"), "/mnt/c/ai_cache")
            or (root / ".cache")
        )
        models_home = (
            _first_existing_dir(os.getenv("EDGE_DEID_MODELS_HOME"), "/mnt/c/ai_models")
            or (root / "models")
        )
        data_home = (
            _first_existing_dir(os.getenv("EDGE_DEID_DATA_HOME"), "/mnt/data")
            or (root / "data")
        )

        return cls(
            project_root=root,
            cache_home=Path(cache_home).expanduser().resolve(),
            models_home=Path(models_home).expanduser().resolve(),
            data_home=Path(data_home).expanduser().resolve(),
        )

    @property
    def datasets_home(self) -> Path:
        return Path(os.getenv("EDGE_DEID_DATASETS_HOME", str(self.data_home / "datasets"))).expanduser().resolve()

    @property
    def training_home(self) -> Path:
        return Path(os.getenv("EDGE_DEID_TRAINING_HOME", str(self.data_home / "training"))).expanduser().resolve()

    @property
    def training_runs_home(self) -> Path:
        return self.training_home / "runs"

    @property
    def training_logs_home(self) -> Path:
        return self.training_home / "logs"

    # --- Project-specific sub-roots ---

    @property
    def edge_deid_models_home(self) -> Path:
        return self.models_home / "detection" / "edge_deid"

    @property
    def edge_deid_datasets_home(self) -> Path:
        return self.datasets_home / "edge_deid"

    @property
    def edge_deid_training_runs_home(self) -> Path:
        return self.training_runs_home / "edge_deid"

