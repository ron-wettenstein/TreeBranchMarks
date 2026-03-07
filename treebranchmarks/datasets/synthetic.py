"""
Synthetic dataset with configurable (n, F) for controlled benchmarking.

Because this dataset is generated from a fixed random seed, it is
reproducible and does not require downloading anything.  It is ideal
for sweeping over F independently of any real dataset's fixed feature count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.dataset import Dataset


class SyntheticDataset(Dataset):
    """
    Synthetic classification dataset generated via sklearn.

    The dataset name encodes the construction parameters so that different
    (n_samples, n_features) configurations have independent caches.

    Parameters
    ----------
    n_samples : int
        Total number of rows.
    n_features : int
        Number of features (= F parameter).
    n_informative : int
        Number of features that are actually informative.
        Defaults to min(n_features, 10).
    n_classes : int
        Number of target classes (default 2 for binary).
    random_state : int
        Random seed for reproducibility.
    cache_root : Path
        Root cache directory.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        n_features: int = 20,
        n_informative: int | None = None,
        n_classes: int = 2,
        random_state: int = 42,
        cache_root: Path = Path("cache"),
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative if n_informative is not None else min(n_features, 10)
        self.n_classes = n_classes
        self.random_state = random_state
        self.cache_root = cache_root

    @property
    def name(self) -> str:  # type: ignore[override]
        return (
            f"synthetic_n{self.n_samples}_f{self.n_features}"
            f"_c{self.n_classes}_seed{self.random_state}"
        )

    def download(self) -> None:
        # Nothing to download; generated in preprocess().
        pass

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        from sklearn.datasets import make_classification

        X_arr, y_arr = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=max(0, self.n_features - self.n_informative - 2),
            n_classes=self.n_classes,
            random_state=self.random_state,
        )
        columns = [f"x{i}" for i in range(self.n_features)]
        X = pd.DataFrame(X_arr.astype(np.float64), columns=columns)
        y = pd.Series(y_arr.astype(np.int64), name="target")
        return X, y

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_informative": self.n_informative,
            "n_classes": self.n_classes,
            "task": "classification",
            "source": "sklearn.datasets.make_classification",
            "random_state": self.random_state,
        }
