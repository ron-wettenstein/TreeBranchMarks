"""
HIGGS dataset (UCI).

  n_samples  : 11,000,000
  n_features : 28
  target     : label (binary classification — signal vs background)
  source     : https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from treebranchmarks.core.dataset import Dataset

_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
_COLS = ["label"] + [f"f{i}" for i in range(28)]


class HIGGSDataset(Dataset):
    """
    HIGGS binary classification dataset from UCI.

    First column is the label; columns f0–f27 are float features.
    Downloaded directly from the UCI repository as a gzipped CSV (~8 GB uncompressed).
    """

    name = "higgs"

    def __init__(self, cache_root: Path = Path("cache")) -> None:
        self.cache_root = cache_root

    def download(self) -> None:
        # The CSV is read directly from the URL in preprocess(); nothing to stage.
        pass

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        print(f"[dataset:{self.name}] Downloading from UCI (this may take a while)…")
        df = pd.read_csv(_URL, header=None, names=_COLS)
        X = df.drop(columns="label")
        y = df["label"].astype(int)
        return X, y

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "n_samples": 11_000_000,
            "n_features": 28,
            "columns": _COLS[1:],
            "target": "label",
            "task": "classification",
            "source": _URL,
        }
