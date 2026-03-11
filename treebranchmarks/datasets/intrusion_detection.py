from treebranchmarks.datasets.google_drive_dataset import GoogleDriveDataset


class IntrusionDetectionDataset(GoogleDriveDataset):
    """
    KDD Cup 1999 Intrusion Detection dataset (hosted on Google Drive).
    Was preprocessed in this notebook:
    https://colab.research.google.com/drive/1AUfFJ0B5SbfJCeu5Nz8y5HdkDxL6UHZ7?usp=drive_link

    target  : target column (regression)
    source  : Google Drive (pre-processed parquet)
                file id: 1sExGIsElOZFEJUv5f42EJzvQZZk8shUg
    """
    name        = "intrusion_detection"
    _FILE_ID    = "1sExGIsElOZFEJUv5f42EJzvQZZk8shUg"
    _FILENAME   = "data.parquet"
    _TARGET_COL = "target"
    _TASK       = "regression"
