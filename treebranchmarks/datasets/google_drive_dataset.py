"""
Shared base class for single-file Google Drive datasets.

Subclasses set four class attributes and nothing else:
  _FILE_ID    : str  — Google Drive file ID
  _FILENAME   : str  — filename to save under _raw_dir()
  _TARGET_COL : str  — column to use as y (dropped from X)
  _TASK       : str  — "classification" or "regression"

All of __init__, download(), preprocess(), and dump_details() are
fully implemented here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.dataset import Dataset


class GoogleDriveDataset(Dataset):
    """Dataset that downloads a single parquet file from Google Drive."""

    _FILE_ID:    str
    _FILENAME:   str
    _TARGET_COL: str
    _TASK:       str

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        import gdown

        dest = self._raw_dir() / self._FILENAME
        if not dest.exists():
            gdown.download(
                f"https://drive.google.com/uc?id={self._FILE_ID}",
                str(dest),
                quiet=False,
            )

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        data = pd.read_parquet(raw_dir / self._FILENAME)
        y = data[self._TARGET_COL].reset_index(drop=True)
        X = data.drop(columns=[self._TARGET_COL]).astype(np.float64).reset_index(drop=True)
        return X, y

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "target": self._TARGET_COL,
            "task": self._TASK,
            "source": "Google Drive (pre-processed parquet)",
        }
