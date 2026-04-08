"""File state cache - tracks read files for edit validation."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class FileState:
    """State of a file when last read."""

    content: str
    mtime: float
    offset: int | None = None
    limit: int | None = None
    is_partial: bool = False


class FileStateCache:
    """LRU cache of recently read file states.

    Used by Edit tool to:
    1. Enforce read-before-edit
    2. Detect stale writes (file changed since last read)
    """

    def __init__(self, max_entries: int = 100, max_bytes: int = 25 * 1024 * 1024):
        self._cache: OrderedDict[str, FileState] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._current_bytes = 0

    def get(self, path: str) -> FileState | None:
        normalized = os.path.normpath(path)
        if normalized in self._cache:
            self._cache.move_to_end(normalized)
            return self._cache[normalized]
        return None

    def set(self, path: str, state: FileState) -> None:
        normalized = os.path.normpath(path)
        content_size = len(state.content.encode())

        # Remove old entry if exists
        if normalized in self._cache:
            old = self._cache.pop(normalized)
            self._current_bytes -= len(old.content.encode())

        # Evict LRU entries if needed
        while self._cache and (
            len(self._cache) >= self._max_entries or self._current_bytes + content_size > self._max_bytes
        ):
            _, evicted = self._cache.popitem(last=False)
            self._current_bytes -= len(evicted.content.encode())

        self._cache[normalized] = state
        self._current_bytes += content_size

    def has_been_read(self, path: str) -> bool:
        return os.path.normpath(path) in self._cache

    def clear(self) -> None:
        self._cache.clear()
        self._current_bytes = 0
