"""
IEEE-CIS Fraud Detection dataset (pre-processed, hosted on Google Drive).

  target  : isFraud (binary classification)
  source  : Google Drive (train + test parquet files, already pre-processed)

Two parquet files are downloaded on first use via gdown and concatenated:
  train : 1pcig33_Xya1ZR8RexLHJwT4lpxLOiqiP
  test  : 1G_YDmllI_q9yNNLsLkO4CgUL3_adnpL5

Requires the `gdown` package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.dataset import Dataset

_TRAIN_FILE_ID = "1pcig33_Xya1ZR8RexLHJwT4lpxLOiqiP"
_TEST_FILE_ID  = "1G_YDmllI_q9yNNLsLkO4CgUL3_adnpL5"


class FraudDetectionDataset(Dataset):
    """
    IEEE-CIS Fraud Detection binary classification dataset.

    Downloads two pre-processed parquet files from Google Drive on first use,
    then concatenates them. All columns except 'isFraud' are features.
    """

    name = "fraud_detection"

    def __init__(self, cache_root: Path = Path("cache"), use_cache: bool = True) -> None:
        self.cache_root = cache_root
        self.use_cache = use_cache

    def download(self) -> None:
        import gdown

        raw_dir = self._raw_dir()
        train_path = raw_dir / "train.parquet"
        test_path  = raw_dir / "test.parquet"

        if not train_path.exists():
            gdown.download(f"https://drive.google.com/uc?id={_TRAIN_FILE_ID}", str(train_path), quiet=False)
        if not test_path.exists():
            gdown.download(f"https://drive.google.com/uc?id={_TEST_FILE_ID}", str(test_path), quiet=False)

    def preprocess(self, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
        train = pd.read_parquet(raw_dir / "train.parquet")
        test  = pd.read_parquet(raw_dir / "test.parquet")
        data  = pd.concat([train, test], ignore_index=True)

        y = data["isFraud"].astype(np.int64)
        X = data.drop(columns=["isFraud"]).astype(np.float64)

        return X, y

    def dump_details(self) -> dict:
        return {
            "name": self.name,
            "target": "isFraud",
            "task": "classification",
            "source": "Google Drive (pre-processed parquet)",
        }
