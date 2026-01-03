"""Storage layout helpers for EdgeDeID Studio.

This package centralizes the filesystem layout used for datasets, models, and caches.
It is primarily intended for dev/training workflows and should remain offline-first.
"""

from .layout import StorageLayout, dataset_slug
from .env import apply_cache_env_defaults, cache_env_defaults

__all__ = [
    "StorageLayout",
    "dataset_slug",
    "apply_cache_env_defaults",
    "cache_env_defaults",
]

