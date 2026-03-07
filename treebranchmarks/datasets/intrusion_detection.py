"""
KDD Cup 1999 Intrusion Detection dataset (hosted on Google Drive).

  target  : target column
  task    : regression
  source  : Google Drive (pre-processed parquet)
            file id: 1YPb6yD3rQDLdhRaQLRmu10x9orC6OJ7c

Downloaded on first use via gdown. Requires the `gdown` package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.dataset import Dataset

_FILE_ID = "1YPb6yD3rQDLdhRaQLRmu10x9orC6OJ7c"


class IntrusionDetectionDataset(Dataset):
    """
    KDD Cup 1999 Intrusion Detection dataset.

    Downloads a pre-processed parquet from Google Drive on first use.
    All columns except 'target' are used as features.
    """

    name = "intrusion_detection"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        import gdown

        raw_dir = self._raw_dir()
        data_path = raw_dir / "data.parquet"

        if not data_path.exists():
            gdown.download(f"https://drive.google.com/uc?id={_FILE_ID}", str(data_path), quiet=False)

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        data = pd.read_parquet(raw_dir / "data.parquet")

        y = data["target"].reset_index(drop=True)
        X = data.drop(columns=["target"]).astype(np.float64).reset_index(drop=True)

        return X, y

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "target": "target",
            "task": "regression",
            "source": "Google Drive (pre-processed parquet)",
        }
