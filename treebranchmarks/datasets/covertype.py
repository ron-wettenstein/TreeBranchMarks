"""
Forest Covertype dataset (sklearn built-in).

  n_samples  : 581,012
  n_features : 54
  target     : forest cover type (7-class classification, int)
  source     : sklearn.datasets.fetch_covtype

This is a good dataset for stressing high-F and high-n regimes.
sklearn downloads it on first access (~74 MB).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.dataset import Dataset


class CovertypeDataset(Dataset):
    """
    Forest Covertype multi-class classification dataset.

    sklearn downloads the raw data on first use.  Our cache layer then
    stores the numpy arrays so subsequent loads are instant.
    """

    name = "covertype"

    def __init__(self, cache_root: Path = Path("cache")) -> None:
        self.cache_root = cache_root

    def download(self) -> None:
        # sklearn handles its own download.
        pass

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        from sklearn.datasets import fetch_covtype

        data = fetch_covtype()
        feature_names = (
            data.feature_names if hasattr(data, "feature_names")
            else [f"x{i}" for i in range(data.data.shape[1])]
        )
        X = pd.DataFrame(data.data, columns=feature_names).astype(np.float64)
        # sklearn returns 1-indexed class labels; keep as-is
        y = pd.Series(data.target, name="cover_type", dtype=np.int64)
        return X, y

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "n_samples": 581_012,
            "n_features": 54,
            "target": "cover_type",
            "n_classes": 7,
            "task": "classification",
            "source": "sklearn.datasets.fetch_covtype",
        }
