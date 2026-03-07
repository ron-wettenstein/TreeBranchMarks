"""
Breast Cancer Wisconsin dataset (sklearn built-in).

  n_samples  : 569
  n_features : 30
  target     : binary classification (malignant / benign)
  source     : sklearn.datasets.load_breast_cancer
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.dataset import Dataset


class BreastCancerDataset(Dataset):
    """
    Breast Cancer Wisconsin classification dataset.

    No download step required — sklearn loads it from a bundled file.
    """

    name = "breast_cancer"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        pass  # sklearn built-in, no download needed

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names).astype(np.float64)
        y = pd.Series(data.target, name="target", dtype=np.int64)
        return X, y

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "n_samples": 569,
            "n_features": 30,
            "target": "malignant/benign",
            "task": "classification",
            "source": "sklearn.datasets.load_breast_cancer",
        }
