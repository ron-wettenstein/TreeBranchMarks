"""
Background (Interventional) SHAP Interaction Values.

Algorithm
---------
Computes pairwise SHAP interaction values using an explicit background dataset
(interventional perturbation).

The shap package does NOT support interventional SHAP interaction values —
its SHAPApproach always returns not_supported.

Woodelf supports this via WoodelfExplainer.shap_interaction_values() with
feature_perturbation="interventional".
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.core.params import TreeParameters
from treebranchmarks.core.task import Task
from treebranchmarks.methods.builtin import SHAP, WOODELF
from woodelf import WoodelfExplainer


# ---------------------------------------------------------------------------
# Approaches
# ---------------------------------------------------------------------------

class SHAPBackgroundInteractionsApproach(Approach):
    """
    SHAP package does not support interventional SHAP interaction values.
    Always returns not_supported.
    """

    name = "shap Package Background SHAP Interactions"
    method = SHAP
    description = "Not supported by the shap package."

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return ApproachOutput(elapsed_s=0.0, not_supported=True)


class WoodelfBackgroundInteractionsApproach(Approach):
    """
    Woodelf interventional SHAP interaction values.

    Requires a background dataset.
    """

    name = "Woodelf Background SHAP Interactions"
    method = WOODELF
    description = "Complexity: O(mTL + nTLD + TL2**D * D**3)."

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

        if X_background is None:
            raise ValueError(
                "WoodelfBackgroundInteractionsApproach requires a background dataset."
            )

        T = trained_model.params.T

        if trained_model.params.D >= self._TREE_LIMIT_DEPTH:
            explainer = WoodelfExplainer(
                trained_model.raw_model,
                data=X_background,
                feature_perturbation="interventional",
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
            data=X_background,
            feature_perturbation="interventional",
        )
        t0 = time.perf_counter()
        explainer.shap_interaction_values(X_explain)
        elapsed = time.perf_counter() - t0
        return ApproachOutput(elapsed_s=elapsed, is_estimated=False)


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def BackgroundSHAPInteractionsTask(
    methods: list | None = None,
    extra_approaches: list[Approach] | None = None,
    n_repeats: int = 3,
    cache_root: Path = Path("cache"),
) -> Task:
    """
    Task for background (interventional) SHAP interaction values.

    Parameters
    ----------
    methods : list[Method] | None
        Which built-in methods to include. Defaults to [SHAP, WOODELF].
    extra_approaches : list[Approach] | None
        Additional Approach instances appended after the built-in ones.
    """
    _registry = {
        SHAP:    SHAPBackgroundInteractionsApproach,
        WOODELF: WoodelfBackgroundInteractionsApproach,
    }
    if methods is None:
        methods = [SHAP, WOODELF]

    approaches: list[Approach] = [_registry[m]() for m in methods if m in _registry]
    if extra_approaches:
        approaches.extend(extra_approaches)

    return Task(
        name="Background SHAP Interactions",
        approaches=approaches,
        n_repeats=n_repeats,
        cache_root=cache_root,
    )
