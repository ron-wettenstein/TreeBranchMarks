"""
LightGBM model wrapper.

Tree parameter extraction
--------------------------
LightGBM exposes the full model structure via:
    model.booster_.dump_model()

This returns a nested dict.  The top-level "tree_info" key contains a list
of per-tree dicts.  Each tree has a "tree_structure" key with the root node.
Leaf nodes have "leaf_value"; internal nodes have "left_child" / "right_child".

T  = len(tree_info)
L  = mean(leaves per tree)
D  = max(depth across all trees)
F  = X.shape[1]
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from treebranchmarks.core.model import ModelConfig, ModelWrapper, TrainedModel
from treebranchmarks.core.params import EnsembleType, TreeParameters, partial_tree_params


def _walk_lgbm_node(node: dict, depth: int = 0) -> tuple[int, int]:
    """
    Recursively walk a LightGBM tree node dict.
    Returns (leaf_count, max_depth).
    """
    if "leaf_value" in node:
        return 1, depth
    leaves = 0
    max_d = depth
    for key in ("left_child", "right_child"):
        child = node.get(key)
        if child is not None:
            c_leaves, c_depth = _walk_lgbm_node(child, depth + 1)
            leaves += c_leaves
            max_d = max(max_d, c_depth)
    return leaves, max_d


class LightGBMWrapper(ModelWrapper):
    """
    Wrapper for LightGBM classifiers and regressors.

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
        import lightgbm as lgb

        hp = {**config.hyperparams, "random_state": config.random_state, "verbose": -1}

        if self.task_type == "regression":
            model = lgb.LGBMRegressor(**hp)
        else:
            n_classes = len(np.unique(y))
            objective = "binary" if n_classes == 2 else "multiclass"
            model = lgb.LGBMClassifier(objective=objective, **hp)

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
        model_dict = raw_model.booster_.dump_model()  # type: ignore[union-attr]
        tree_info = model_dict.get("tree_info", [])

        T = len(tree_info)
        leaf_counts = []
        depth_values = []

        for tree in tree_info:
            root = tree.get("tree_structure", {})
            leaves, max_depth = _walk_lgbm_node(root)
            leaf_counts.append(leaves)
            depth_values.append(max_depth)

        L = float(np.mean(leaf_counts)) if leaf_counts else 1.0
        D = int(max(depth_values)) if depth_values else 0
        F = X.shape[1]

        return partial_tree_params(T=T, D=D, L=L, F=F, ensemble_type=EnsembleType.LIGHTGBM)
