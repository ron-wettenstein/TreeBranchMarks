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
from treebranchmarks.methods.builtin import SHAP, WOODELF
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

    name = "shap Package Path-Dependent SHAP"
    method = SHAP
    description = (
        "Complexity: O(nTLD²)."
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

    name = "Woodelf Path-Dependent SHAP"
    method = WOODELF
    description = (
        "Complexity: O(nTLD + TL2**D * D²)."
    )

    MAX_SUPPORTED_DEPTH = 21

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        import shap

        if trained_model.params.D > self.MAX_SUPPORTED_DEPTH:
            return ApproachOutput(elapsed_s=0.0, not_supported=True)

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
    methods: list | None = None,
    extra_approaches: list[Approach] | None = None,
    n_repeats: int = 3,
    cache_root: Path = Path("cache"),
) -> Task:
    """
    Convenience factory that returns a Task preconfigured for path-dependent SHAP.

    Parameters
    ----------
    methods : list[Method] | None
        Which built-in methods to include.  Defaults to [SHAP, WOODELF].
        Pass a subset to run only selected methods, e.g. ``methods=[SHAP]``.
    extra_approaches : list[Approach] | None
        Additional Approach instances (e.g. a custom method) appended after
        the built-in ones.
    """
    _registry = {
        SHAP:    SHAPTreePathDependentApproach,
        WOODELF: WoodelfSHAPTreePathDependentApproach,
    }
    if methods is None:
        methods = [SHAP, WOODELF]

    approaches: list[Approach] = [_registry[m]() for m in methods if m in _registry]
    if extra_approaches:
        approaches.extend(extra_approaches)

    return Task(
        name="Path-Dependent SHAP",
        approaches=approaches,
        n_repeats=n_repeats,
        cache_root=cache_root,
    )
