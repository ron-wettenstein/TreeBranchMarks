"""
XGBoost model wrapper.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.model import ModelConfig, ModelWrapper, TrainedModel


class XGBoostWrapper(ModelWrapper):
    """
    Wrapper for XGBoost classifiers and regressors.

    Supports both XGBClassifier and XGBRegressor via the task_type parameter.

    Parameters
    ----------
    task_type : str
        "classification" (default) or "regression".
    """

    def __init__(self, task_type: str = "classification", use_cache: bool = True) -> None:
        super().__init__(use_cache=use_cache)
        self.task_type = task_type

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: ModelConfig,
        dataset_name: str,
    ) -> TrainedModel:
        import xgboost as xgb

        hp = {**config.hyperparams, "random_state": config.random_state}

        if self.task_type == "regression":
            model = xgb.XGBRegressor(**hp)
        else:
            n_classes = len(np.unique(y))
            if n_classes == 2:
                model = xgb.XGBClassifier(**hp)
            else:
                model = xgb.XGBClassifier(num_class=n_classes, **hp)

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
        # XGBoost's built-in JSON serialization preserves the full booster state.
        raw_model.save_model(str(model_dir / "model.json"))  # type: ignore[union-attr]

    def _load_model_artifact(self, model_dir: Path) -> object:
        import xgboost as xgb

        model = (
            xgb.XGBRegressor() if self.task_type == "regression"
            else xgb.XGBClassifier()
        )
        model.load_model(str(model_dir / "model.json"))
        return model

