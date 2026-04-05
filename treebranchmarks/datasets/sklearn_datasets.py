"""
Sklearn built-in datasets.

Classes
-------
BreastCancerDataset      — binary classification   (569 rows,   30 features)
DiabetesDataset          — regression              (442 rows,   10 features)
DigitsDataset            — multiclass (10 classes) (1797 rows,  64 features)
CaliforniaHousingDataset — regression              (20640 rows,  8 features)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.dataset import Dataset


class BreastCancerDataset(Dataset):
    """Breast Cancer Wisconsin binary classification dataset."""

    name = "breast_cancer"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        pass

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


class DiabetesDataset(Dataset):
    """Diabetes regression dataset."""

    name = "diabetes"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        pass

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        from sklearn.datasets import load_diabetes

        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names).astype(np.float64)
        y = pd.Series(data.target, name="target", dtype=np.float64)
        return X, y

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "n_samples": 442,
            "n_features": 10,
            "target": "disease progression",
            "task": "regression",
            "source": "sklearn.datasets.load_diabetes",
        }


class DigitsDataset(Dataset):
    """Digits multiclass classification dataset (10 digit classes)."""

    name = "digits"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        pass

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        from sklearn.datasets import load_digits

        data = load_digits()
        columns = [f"pixel_{i}" for i in range(data.data.shape[1])]
        X = pd.DataFrame(data.data.astype(np.float64), columns=columns)
        y = pd.Series(data.target.astype(np.int64), name="target")
        return X, y

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "n_samples": 1797,
            "n_features": 64,
            "n_classes": 10,
            "target": "digit (0-9)",
            "task": "classification",
            "source": "sklearn.datasets.load_digits",
        }


class CaliforniaHousingDataset(Dataset):
    """California Housing prices regression dataset."""

    name = "california_housing"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
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
