"""
HIGGS dataset (Google Drive parquet).

  n_samples  : 11,000,000
  n_features : 28
  target     : label (binary classification — signal vs background)
  source     : https://drive.google.com/file/d/1iDX-Uxs4SruwpjoKjDrn9DW80wkFid83
"""

from __future__ import annotations

from pathlib import Path

from treebranchmarks.datasets.google_drive_dataset import GoogleDriveDataset


class HIGGSDataset(GoogleDriveDataset):
    """
    HIGGS binary classification dataset, loaded from a Google Drive parquet file.

    First column is the label; columns f0–f27 are float features.

    Was preprocessed in this notebook:
    https://colab.research.google.com/drive/1AUfFJ0B5SbfJCeu5Nz8y5HdkDxL6UHZ7?usp=drive_link

    The full preprocessing code is:

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

    # UCI file format: first column is label, then 28 float features.
    cols = ["label"] + [f"f{i}" for i in range(28)]
    df_higgs = pd.read_csv(URL, header=None, names=cols)

    df_higgs.to_parquet('drive/MyDrive/FILE_LOCATION/higgs_data.parquet')
    """

    name        = "higgs"
    _FILE_ID    = "1iDX-Uxs4SruwpjoKjDrn9DW80wkFid83"
    _FILENAME   = "higgs.parquet"
    _TARGET_COL = "label"
    _TASK       = "classification"
