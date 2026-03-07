"""
California Housing dataset (sklearn built-in).

  n_samples  : 20,640
  n_features : 8
  target     : median house value (regression, float)
  source     : sklearn.datasets.fetch_california_housing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.dataset import Dataset


class CaliforniaHousingDataset(Dataset):
    """
    California Housing prices regression dataset.

    No download step required — sklearn fetches and caches it internally.
    The preprocessed arrays are still cached by the base class to avoid
    repeated sklearn overhead and ensure a consistent array layout.
    """

    name = "california_housing"

    def __init__(self, cache_root: Path = Path("cache")) -> None:
        self.cache_root = cache_root

    def download(self) -> None:
        # sklearn handles its own download; nothing to do here.
        pass

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names).astype(np.float64)
        y = pd.Series(data.target, name="MedHouseVal", dtype=np.float64)
        return X, y

    def dump_details(self) -> dict:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        return {
            "name": self.name,
            "n_samples": 20_640,
            "n_features": 8,
            "feature_names": list(data.feature_names),
            "target": "median_house_value",
            "task": "regression",
            "source": "sklearn.datasets.fetch_california_housing",
        }
