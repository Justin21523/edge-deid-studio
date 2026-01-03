from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Optional, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class ReplacementCache(Generic[K, V]):
    """A small thread-safe LRU cache for deterministic replacements."""

    def __init__(self, max_size: int = 10_000) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self._max_size = int(max_size)
        self._lock = threading.Lock()
        self._data: "OrderedDict[K, V]" = OrderedDict()

    @property
    def max_size(self) -> int:
        return self._max_size

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key not in self._data:
                return None
            value = self._data.pop(key)
            self._data[key] = value
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            self._data[key] = value
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def get_or_set(self, key: K, factory: Callable[[], V]) -> V:
        existing = self.get(key)
        if existing is not None:
            return existing

        value = factory()
        self.set(key, value)
        return value


DEFAULT_REPLACEMENT_CACHE: ReplacementCache[str, str] = ReplacementCache(max_size=10_000)

