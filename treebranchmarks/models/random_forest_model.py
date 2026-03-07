"""
scikit-learn Random Forest wrapper.

Tree parameter extraction
--------------------------
sklearn exposes the internal tree structure via the Cython Tree object
on each estimator:

    estimator.tree_.n_node_samples  — array of sample counts per node
    estimator.tree_.children_left   — left child index (-1 = leaf)
    estimator.tree_.max_depth       — max depth of this tree
    estimator.tree_.n_leaves        — number of leaf nodes

T  = len(estimators_)
L  = mean(estimator.tree_.n_leaves for estimator in estimators_)
D  = max(estimator.tree_.max_depth for estimator in estimators_)
F  = X.shape[1]

Note: sklearn's max_depth attribute on the tree object is the actual depth
of the fitted tree, not the hyperparameter (which can be None = unbounded).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.model import ModelConfig, ModelWrapper, TrainedModel
from treebranchmarks.core.params import EnsembleType, TreeParameters, partial_tree_params


class RandomForestWrapper(ModelWrapper):
    """
    Wrapper for sklearn RandomForestClassifier and RandomForestRegressor.

    Parameters
    ----------
    task_type : str
        "classification" (default) or "regression".
    """

    def __init__(self, task_type: str = "classification") -> None:
        self.task_type = task_type

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: ModelConfig,
        dataset_name: str,
    ) -> TrainedModel:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        hp = {**config.hyperparams, "random_state": config.random_state}

        if self.task_type == "regression":
            model = RandomForestRegressor(**hp)
        else:
            model = RandomForestClassifier(**hp)

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
        # sklearn has no library-native text format; joblib is sklearn's own
        # recommended serialization (it handles numpy arrays efficiently).
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
        estimators = raw_model.estimators_  # type: ignore[union-attr]

        T = len(estimators)
        leaf_counts = [est.tree_.n_leaves for est in estimators]
        depth_values = [est.tree_.max_depth for est in estimators]

        L = float(np.mean(leaf_counts))
        D = int(max(depth_values))
        F = X.shape[1]

        return partial_tree_params(T=T, D=D, L=L, F=F, ensemble_type=EnsembleType.RANDOM_FOREST)
