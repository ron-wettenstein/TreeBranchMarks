"""
Lightweight cache utilities used across the framework.

The heavy caching logic lives in Dataset and ModelWrapper themselves.
This module provides:

  - stable_hash()    — deterministic MD5 of any JSON-serializable object
  - CacheStore       — optional generic key-value cache backed by joblib,
                       for cases where you want to cache arbitrary objects
                       without subclassing Dataset or ModelWrapper

CacheStore is intentionally simple.  You do not need to use it unless you
are adding custom cacheable objects outside the built-in wrappers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, TypeVar

T = TypeVar("T")


def stable_hash(obj: Any) -> str:
    """
    Return a deterministic MD5 hex digest of a JSON-serializable object.

    The object is serialized with sorted keys so that dict ordering does
    not affect the hash.
    """
    serialized = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


class CacheStore:
    """
    Generic key → object cache backed by joblib on disk.

    Keys are arbitrary strings.  Values are serialized with joblib, which
    handles numpy arrays efficiently (compressed by default).

    Parameters
    ----------
    root : Path
        Cache directory.  Created automatically if it does not exist.
    namespace : str
        Subdirectory under root for this store's entries.
    compress : int
        joblib compression level (0–9).  Default 3.
    """

    def __init__(
        self,
        root: Path = Path("cache"),
        namespace: str = "generic",
        compress: int = 3,
    ) -> None:
        self.root = root
        self.namespace = namespace
        self.compress = compress
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def _dir(self) -> Path:
        return self.root / self.namespace

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for key, or None if not present."""
        path = self._path(key)
        if not path.exists():
            return None
        import joblib
        return joblib.load(path)

    def put(self, key: str, value: Any) -> None:
        """Store value under key."""
        import joblib
        joblib.dump(value, self._path(key), compress=self.compress)

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        """Delete all entries in this namespace."""
        import shutil
        shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Use a hash of the key as the filename to avoid filesystem issues
        # with long or special-character keys.
        filename = stable_hash(key) + ".joblib"
        return self._dir / filename
