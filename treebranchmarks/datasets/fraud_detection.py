from treebranchmarks.datasets.google_drive_dataset import GoogleDriveDataset


class FraudDetectionDataset(GoogleDriveDataset):
    """
    IEEE-CIS Fraud Detection dataset (pre-processed, hosted on Google Drive).
    Was preprocessed in this notebook:
    https://colab.research.google.com/drive/1AUfFJ0B5SbfJCeu5Nz8y5HdkDxL6UHZ7?usp=drive_link

    target  : isFraud (binary classification)
    source  : Google Drive (pre-processed parquet)
                file id: 1A1Qdtron9XtZ6h85uNdaFCprUkH5KA5P
    """
    name        = "fraud_detection"
    _FILE_ID    = "1A1Qdtron9XtZ6h85uNdaFCprUkH5KA5P"
    _FILENAME   = "data.parquet"
    _TARGET_COL = "isFraud"
    _TASK       = "classification"
