"""
Path-Dependent SHAP Interaction Values.

Algorithm
---------
Computes pairwise SHAP interaction values using the tree path-dependent
perturbation (no background dataset required).

Reference implementation: shap.TreeExplainer with
    feature_perturbation="tree_path_dependent"
    calling shap_interaction_values().

Woodelf: WoodelfExplainer with feature_perturbation="tree_path_dependent"
    calling shap_interaction_values().
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import shap

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.core.params import TreeParameters
from treebranchmarks.core.task import Task
from treebranchmarks.methods.builtin import SHAP, WOODELF
from woodelf import WoodelfExplainer


# ---------------------------------------------------------------------------
# Approaches
# ---------------------------------------------------------------------------

class SHAPPathDependentInteractionsApproach(Approach):
    """
    SHAP TreeExplainer path-dependent interaction values.

    No background dataset needed.
    """

    name = "shap Package Path-Dependent SHAP Interactions"
    method = SHAP
    description = "Complexity: O(nTLFD²)"

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
        explainer.shap_interaction_values(X_explain)
        elapsed = time.perf_counter() - t0

        return ApproachOutput(elapsed_s=elapsed)


class WoodelfPathDependentInteractionsApproach(Approach):
    """
    Woodelf path-dependent SHAP interaction values.

    No background dataset needed.
    """

    name = "Woodelf Path-Dependent SHAP Interactions"
    method = WOODELF
    description = "Complexity: O(nTLD + TL2**D * D**3)."

    MAX_SUPPORTED_DEPTH = 18
    _TREE_LIMIT_DEPTH   = 15

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        if trained_model.params.D > self.MAX_SUPPORTED_DEPTH:
            return ApproachOutput(elapsed_s=0.0, memory_crash=True)

        T = trained_model.params.T

        if trained_model.params.D >= self._TREE_LIMIT_DEPTH:
            explainer = WoodelfExplainer(
                trained_model.raw_model,
                feature_perturbation="tree_path_dependent",
            )
            t0 = time.perf_counter()
            explainer.shap_interaction_values(X_explain, tree_limit=1)
            elapsed = time.perf_counter() - t0
            return ApproachOutput(
                elapsed_s=elapsed * T,
                is_estimated=True,
                estimation_description=(
                    f"D={trained_model.params.D} ≥ {self._TREE_LIMIT_DEPTH}: "
                    f"ran with tree_limit=1, extrapolated ×{T} trees"
                ),
            )

        explainer = WoodelfExplainer(
            trained_model.raw_model,
            feature_perturbation="tree_path_dependent",
        )
        t0 = time.perf_counter()
        explainer.shap_interaction_values(X_explain)
        elapsed = time.perf_counter() - t0

        return ApproachOutput(elapsed_s=elapsed)


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def PathDependentInteractionsTask(
    methods: list | None = None,
    extra_approaches: list[Approach] | None = None,
    n_repeats: int = 3,
    cache_root: Path = Path("cache"),
) -> Task:
    """
    Task for path-dependent SHAP interaction values.

    Parameters
    ----------
    methods : list[Method] | None
        Which built-in methods to include. Defaults to [SHAP, WOODELF].
    extra_approaches : list[Approach] | None
        Additional Approach instances appended after the built-in ones.
    """
    _registry = {
        SHAP:    SHAPPathDependentInteractionsApproach,
        WOODELF: WoodelfPathDependentInteractionsApproach,
    }
    if methods is None:
        methods = [SHAP, WOODELF]

    approaches: list[Approach] = [_registry[m]() for m in methods if m in _registry]
    if extra_approaches:
        approaches.extend(extra_approaches)

    return Task(
        name="Path-Dependent SHAP Interactions",
        approaches=approaches,
        n_repeats=n_repeats,
        cache_root=cache_root,
    )
