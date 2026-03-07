"""
Abstract base class for benchmark datasets.

Lifecycle
---------
1. `load()` is the single public entry point.
2. On first call it runs: download() → preprocess() → _save_cache()
3. On subsequent calls it skips straight to _load_cache().
4. `dump_details()` returns a metadata dict that is stored alongside results.

Subclasses must implement `download()` and `preprocess()`.
The cache is stored under:
    {cache_root}/datasets/{name}/X.parquet   — feature DataFrame (preserves column names)
    {cache_root}/datasets/{name}/y.npy       — target array
    {cache_root}/datasets/{name}/details.json
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class Dataset(ABC):
    """
    Base class for benchmark datasets.

    Attributes
    ----------
    name : str
        Unique, filesystem-safe identifier (e.g. "california_housing").
    cache_root : Path
        Root cache directory. Defaults to <project root>/cache.
        Override per instance to redirect caching.
    """

    name: str
    cache_root: Path = Path("cache")
    use_cache: bool = True  # set False for small/fast datasets to skip disk I/O

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return (X, y) as a DataFrame and a Series.

        If use_cache=True (default), reads from / writes to the on-disk cache.
        If use_cache=False, always downloads and preprocesses without caching.
        """
        if self.use_cache:
            if self._is_cached():
                return self._load_cache()
            print(f"[dataset:{self.name}] Cache miss — downloading and preprocessing.")
            self.download()
            X, y = self.preprocess(self._raw_dir())
            self._save_cache(X, y)
            print(f"[dataset:{self.name}] Cached {X.shape[0]} rows × {X.shape[1]} features.")
            return X, y

        self.download()
        return self.preprocess(self._raw_dir())

    def dump_details(self) -> dict:
        """
        Return metadata about this dataset.

        The base implementation returns shape info from the cached arrays.
        Subclasses should override to include source URL, feature names, etc.
        """
        details_path = self._cache_dir() / "details.json"
        if details_path.exists():
            with open(details_path) as f:
                return json.load(f)
        X, y = self.load()
        return {"name": self.name, "n_samples": X.shape[0], "n_features": X.shape[1]}

    def invalidate_cache(self) -> None:
        """Delete cached arrays, forcing a fresh download on next load()."""
        import shutil
        if self._cache_dir().exists():
            shutil.rmtree(self._cache_dir())
            print(f"[dataset:{self.name}] Cache cleared.")

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement these two methods
    # ------------------------------------------------------------------

    @abstractmethod
    def download(self) -> None:
        """
        Download raw source data into self._raw_dir().

        If the source is a sklearn built-in or fully synthetic, this method
        may do nothing (the data is generated in preprocess() instead).
        """

    @abstractmethod
    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        """
        Transform raw files in raw_dir into a (X, y) pair.

        X : pd.DataFrame, shape (n_samples, n_features), dtype float64
            Column names should be meaningful feature names when available.
        y : pd.Series,    shape (n_samples,), dtype float64 or int64
        """

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_dir(self) -> Path:
        return self.cache_root / "datasets" / self.name

    def _raw_dir(self) -> Path:
        d = self._cache_dir() / "raw"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _is_cached(self) -> bool:
        return (
            (self._cache_dir() / "X.parquet").exists()
            and (self._cache_dir() / "y.npy").exists()
        )

    def _save_cache(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._cache_dir().mkdir(parents=True, exist_ok=True)
        X.to_parquet(self._cache_dir() / "X.parquet", index=False)
        np.save(self._cache_dir() / "y.npy", y.to_numpy())
        try:
            details = self.dump_details()
        except Exception:
            details = self._build_details(X, y)
        with open(self._cache_dir() / "details.json", "w") as f:
            json.dump(details, f, indent=2)

    def _build_details(self, X: pd.DataFrame, y: pd.Series) -> dict:
        return {
            "name": self.name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "columns": list(X.columns),
        }

    def _load_cache(self) -> tuple[pd.DataFrame, pd.Series]:
        X = pd.read_parquet(self._cache_dir() / "X.parquet")
        y = pd.Series(np.load(self._cache_dir() / "y.npy"), name="y")
        return X, y
