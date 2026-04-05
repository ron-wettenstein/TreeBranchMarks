"""
Synthetic workload datasets matching the QuadratureSHAP benchmark suite.

All three datasets use n=40000 rows and F=50 features, generated from a
fixed random seed so they are reproducible.

  EasyLinearDataset     — binary classification, linearly separable signal
  RandomLabelsDataset   — binary classification, fully random labels (hard trees)
  RandomRegressionDataset — regression, fully random targets (hard trees)

The "hard" datasets (random labels/regression) produce very deep, bushy trees
that stress-test SHAP implementations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.dataset import Dataset

_N = 40_000
_F = 50
_SEED = 20260320


class EasyLinearDataset(Dataset):
    """
    Binary classification dataset with a simple linear decision boundary.

    y = 1  iff  X[:,0] + 0.5*X[:,1] - 0.25*X[:,2] > 0
    """

    name = "easy_linear"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        pass

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(_SEED)
        X_arr = rng.standard_normal((_N, _F)).astype(np.float32)
        y_arr = (X_arr[:, 0] + 0.5 * X_arr[:, 1] - 0.25 * X_arr[:, 2] > 0).astype(np.int64)
        columns = [f"x{i}" for i in range(_F)]
        return pd.DataFrame(X_arr.astype(np.float64), columns=columns), pd.Series(y_arr, name="target")

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "n_samples": _N,
            "n_features": _F,
            "task": "classification",
            "source": "synthetic linear boundary",
            "random_state": _SEED,
        }


class RandomLabelsDataset(Dataset):
    """
    Binary classification dataset with fully random labels.

    Produces very deep, bushy trees — stress-tests SHAP algorithms.
    """

    name = "random_labels"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        pass

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(_SEED)
        X_arr = rng.standard_normal((_N, _F)).astype(np.float32)
        y_arr = rng.integers(0, 2, size=_N, dtype=np.int64)
        columns = [f"x{i}" for i in range(_F)]
        return pd.DataFrame(X_arr.astype(np.float64), columns=columns), pd.Series(y_arr, name="target")

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "n_samples": _N,
            "n_features": _F,
            "task": "classification",
            "source": "synthetic random labels",
            "random_state": _SEED,
        }


class RandomRegressionDataset(Dataset):
    """
    Regression dataset with fully random targets.

    Produces very deep, bushy trees — stress-tests SHAP algorithms.
    """

    name = "random_regression"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        pass

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(_SEED)
        X_arr = rng.standard_normal((_N, _F)).astype(np.float32)
        y_arr = rng.standard_normal(_N).astype(np.float64)
        columns = [f"x{i}" for i in range(_F)]
        return pd.DataFrame(X_arr.astype(np.float64), columns=columns), pd.Series(y_arr, name="target")

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "n_samples": _N,
            "n_features": _F,
            "task": "regression",
            "source": "synthetic random regression",
            "random_state": _SEED,
        }
