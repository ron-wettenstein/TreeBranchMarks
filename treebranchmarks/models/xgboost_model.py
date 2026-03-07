"""
XGBoost model wrapper.

Tree parameter extraction
--------------------------
XGBoost exposes the raw tree structure via:
    booster.get_dump(dump_format='json')

This returns a list of JSON strings, one per tree.  Each tree is a nested
dict where leaf nodes have a "leaf" key and internal nodes have "children".
We walk every tree to count leaves and compute max depth.

T  = len(trees)
L  = mean(leaves per tree)
D  = max(depth across all trees)
F  = X.shape[1]  (feature count from training data)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.model import ModelConfig, ModelWrapper, TrainedModel
from treebranchmarks.core.params import EnsembleType, TreeParameters, partial_tree_params


def _walk_xgb_tree(node: dict, depth: int = 0) -> tuple[int, int]:
    """
    Recursively walk an XGBoost tree node dict.
    Returns (leaf_count, max_depth).
    """
    if "leaf" in node:
        return 1, depth
    leaves = 0
    max_d = depth
    for child in node.get("children", []):
        c_leaves, c_depth = _walk_xgb_tree(child, depth + 1)
        leaves += c_leaves
        max_d = max(max_d, c_depth)
    return leaves, max_d


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

    def _extract_tree_params(
        self,
        raw_model: object,
        X: pd.DataFrame,
        config: ModelConfig,
    ) -> TreeParameters:
        import xgboost as xgb

        booster = raw_model.get_booster()  # type: ignore[union-attr]
        tree_dumps = booster.get_dump(dump_format="json")

        T = len(tree_dumps)
        leaf_counts = []
        depth_values = []

        for tree_str in tree_dumps:
            root = json.loads(tree_str)
            leaves, max_depth = _walk_xgb_tree(root)
            leaf_counts.append(leaves)
            depth_values.append(max_depth)

        L = float(np.mean(leaf_counts)) if leaf_counts else 1.0
        D = int(max(depth_values)) if depth_values else 0
        F = X.shape[1]

        return partial_tree_params(T=T, D=D, L=L, F=F, ensemble_type=EnsembleType.XGBOOST)
