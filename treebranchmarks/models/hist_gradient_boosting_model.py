"""
scikit-learn HistGradientBoosting model wrapper.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from treebranchmarks.core.model import ModelConfig, ModelWrapper, TrainedModel


class HistGradientBoostingWrapper(ModelWrapper):
    """
    Wrapper for sklearn HistGradientBoostingClassifier and HistGradientBoostingRegressor.

    Parameters
    ----------
    task_type : str
        "classification" (default) or "regression".
    """

    def __init__(self, task_type: str = "regression", use_cache: bool = True) -> None:
        super().__init__(use_cache=use_cache)
        self.task_type = task_type

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: ModelConfig,
        dataset_name: str,
    ) -> TrainedModel:
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        hp = {**config.hyperparams, "random_state": config.random_state}

        if self.task_type == "regression":
            model = HistGradientBoostingRegressor(**hp)
        else:
            model = HistGradientBoostingClassifier(**hp)

        t0 = time.perf_counter()
        model.fit(X, y)
        train_time = time.perf_counter() - t0

        params = self._extract_tree_params(model, X, config)
        return TrainedModel(
            raw_model=model,
            config=config,
            params=params,
            train_time_s=train_time,
            dataset_name=dataset_name,
        )

    def _save_model_artifact(self, model_dir: Path, raw_model: object) -> None:
        import joblib
        joblib.dump(raw_model, model_dir / "model.joblib")

    def _load_model_artifact(self, model_dir: Path) -> object:
        import joblib
        return joblib.load(model_dir / "model.joblib")

