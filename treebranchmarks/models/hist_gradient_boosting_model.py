"""
scikit-learn HistGradientBoosting model wrapper.

Tree parameter extraction
--------------------------
HistGradientBoosting stores its fitted trees in:
    model._predictors  — list[list[TreePredictor]]
                         outer: one entry per boosting iteration
                         inner: one TreePredictor per class (length 1 for regression)

Each TreePredictor has a `nodes` structured numpy array with fields:
    is_leaf  — bool
    depth    — uint32

T  = total number of TreePredictor objects (== n_iterations for regression)
L  = mean(nodes['is_leaf'].sum() for each predictor)
D  = max(nodes['depth'].max() for each predictor)
F  = X.shape[1]
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.model import ModelConfig, ModelWrapper, TrainedModel
from treebranchmarks.core.params import EnsembleType, TreeParameters, partial_tree_params


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

    def _extract_tree_params(
        self,
        raw_model: object,
        X: pd.DataFrame,
        config: ModelConfig,
    ) -> TreeParameters:
        # Flatten all TreePredictor objects across iterations and classes.
        predictors = [p for plist in raw_model._predictors for p in plist]  # type: ignore[union-attr]

        T = len(predictors)
        leaf_counts = [int(p.nodes["is_leaf"].sum()) for p in predictors]
        depth_values = [int(p.nodes["depth"].max()) for p in predictors]

        L = float(np.mean(leaf_counts)) if leaf_counts else 1.0
        D = int(max(depth_values)) if depth_values else 0
        F = X.shape[1]

        return partial_tree_params(
            T=T, D=D, L=L, F=F,
            ensemble_type=EnsembleType.HIST_GRADIENT_BOOSTING,
        )
