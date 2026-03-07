"""
Path-Dependent SHAP (TreeSHAP, interventional=False).

Algorithm
---------
Uses the decision path through the tree to compute SHAP values without a
background dataset.  The data distribution is estimated from the training
data that passes through each internal node.

Theoretical complexity: O(T * L^2 * n)
  - T trees, each with L leaves
  - For each of the n rows, the algorithm walks the tree and at each split
    considers which leaves are reachable from each ancestor

Reference implementation: shap.TreeExplainer with
    feature_perturbation="tree_path_dependent"
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.core.params import TreeParameters
from treebranchmarks.core.task import Task
from woodelf import WoodelfExplainer
import shap


# ---------------------------------------------------------------------------
# Approach
# ---------------------------------------------------------------------------

class SHAPTreePathDependentApproach(Approach):
    """
    SHAP TreeExplainer with feature_perturbation="tree_path_dependent".

    This approach does not require a background dataset.  X_background is
    ignored.

    Complexity: O(T * L^2 * n)
    """

    name = "shap_tree_path_dependent"
    method = "shap"
    description = (
        "SHAP TreeExplainer (tree_path_dependent). "
        "No background dataset needed. Complexity: O(T * L² * n)."
    )

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        explainer = shap.TreeExplainer(
            trained_model.raw_model,
            feature_perturbation="tree_path_dependent",
        )
        t0 = time.perf_counter()
        explainer.shap_values(X_explain, check_additivity=False)
        elapsed = time.perf_counter() - t0

        return ApproachOutput(elapsed_s=elapsed)

    def complexity_formula(self, params: TreeParameters) -> Optional[float]:
        # O(T * L^2 * n)
        return params.T * (params.L ** 2) * params.n




class WoodelfSHAPTreePathDependentApproach(Approach):
    """
    SHAP TreeExplainer with feature_perturbation="tree_path_dependent".

    This approach does not require a background dataset.  X_background is
    ignored.

    Complexity: O(T * L^2 * n)
    """

    name = "woodelf_shap_tree_path_dependent"
    method = "woodelf"
    description = (
        "WoodelfExplainer (tree_path_dependent). "
        "No background dataset needed. "
    )

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        import shap

        explainer = WoodelfExplainer(
            trained_model.raw_model,
            feature_perturbation="tree_path_dependent",
        )
        t0 = time.perf_counter()
        explainer.shap_values(X_explain)
        elapsed = time.perf_counter() - t0

        return ApproachOutput(elapsed_s=elapsed)

    def complexity_formula(self, params: TreeParameters) -> Optional[float]:
        return params.n * params.T * params.D  + params.T * params.L * (2 ** params.D)* (params.D ** 2)

# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def PathDependentSHAPTask(
    extra_approaches: list[Approach] | None = None,
    n_repeats: int = 3,
    cache_root: Path = Path("cache"),
) -> Task:
    """
    Convenience factory that returns a Task preconfigured for path-dependent SHAP.

    Pass extra_approaches to include your own algorithm implementations
    alongside the reference shap library approach.
    """
    approaches: list[Approach] = [SHAPTreePathDependentApproach(), WoodelfSHAPTreePathDependentApproach()]
    if extra_approaches:
        approaches.extend(extra_approaches)

    return Task(
        name="path_dependent_shap",
        approaches=approaches,
        n_repeats=n_repeats,
        cache_root=cache_root,
    )
