"""
scikit-learn Decision Tree wrapper (single tree, T=1).

Tree parameter extraction
--------------------------
    estimator.tree_.n_leaves   — number of leaf nodes
    estimator.tree_.max_depth  — actual depth of the fitted tree
    X.shape[1]                 — feature count

T = 1 (always — this is a single tree, not an ensemble)
L = estimator.tree_.n_leaves
D = estimator.tree_.max_depth
F = X.shape[1]
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.model import ModelConfig, ModelWrapper, TrainedModel
from treebranchmarks.core.params import EnsembleType, TreeParameters, partial_tree_params


class DecisionTreeWrapper(ModelWrapper):
    """
    Wrapper for sklearn DecisionTreeClassifier and DecisionTreeRegressor.

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
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        hp = {**config.hyperparams, "random_state": config.random_state}

        if self.task_type == "regression":
            model = DecisionTreeRegressor(**hp)
        else:
            model = DecisionTreeClassifier(**hp)

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
        tree = raw_model.tree_  # type: ignore[union-attr]

        T = 1
        L = float(tree.n_leaves)
        D = int(tree.max_depth)
        F = X.shape[1]

        return partial_tree_params(T=T, D=D, L=L, F=F, ensemble_type=EnsembleType.DECISION_TREE)
