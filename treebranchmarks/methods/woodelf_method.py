"""
WoodelfApproach — WoodelfExplainer covering all 4 task types.

For high depth (D >= 15) uses tree_limit=1 + extrapolation.
For very high depth (D > 21 for PD, D > 18 for BG/interactions) returns memory_crash.

Complexity formulas
-------------------
- path_dependent_shap/interactions : O(nTLD + TL·2^D·D²) / O(nTLD + TL·2^D·D³)
- background_shap/interactions     : O(mTL + nTLD + TL·2^D·D²) / O(mTL + nTLD + TL·2^D·D³)
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.core.params import TreeParameters
from treebranchmarks.methods.builtin import WOODELF
from woodelf import WoodelfExplainer


# ---------------------------------------------------------------------------
# WoodelfApproach
# ---------------------------------------------------------------------------

class WoodelfApproach(Approach):
    """WoodelfExplainer implementation covering all 4 task types."""

    name = "Woodelf"
    method = WOODELF
    description = "Woodelf TreeExplainer implementation."

    _PD_MAX_SUPPORTED_DEPTH = 21   # path-dependent tasks
    _BG_MAX_SUPPORTED_DEPTH = 18   # background / interaction tasks
    _TREE_LIMIT_DEPTH       = 15   # use tree_limit=1 + extrapolation when D >= this

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        max_depth: int,
        feature_perturbation: str,
        use_interactions: bool,
    ) -> ApproachOutput:
        if trained_model.params.D > max_depth:
            return ApproachOutput(elapsed_s=0.0, memory_crash=True)

        if feature_perturbation == "interventional":
            if X_background is None:
                raise ValueError("WoodelfApproach interventional tasks require X_background.")
            explainer = WoodelfExplainer(
                trained_model.raw_model,
                data=X_background,
                feature_perturbation="interventional",
            )
        else:
            explainer = WoodelfExplainer(
                trained_model.raw_model,
                feature_perturbation="tree_path_dependent",
            )

        run_fn = explainer.shap_interaction_values if use_interactions else explainer.shap_values
        T = trained_model.params.T

        if trained_model.params.D >= self._TREE_LIMIT_DEPTH:
            t0 = time.perf_counter()
            run_fn(X_explain, tree_limit=1)
            elapsed = time.perf_counter() - t0
            return ApproachOutput(
                elapsed_s=elapsed * T,
                is_estimated=True,
                estimation_description=(
                    f"D={trained_model.params.D} ≥ {self._TREE_LIMIT_DEPTH}: "
                    f"ran with tree_limit=1, extrapolated ×{T} trees"
                ),
            )

        t0 = time.perf_counter()
        run_fn(X_explain)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Task methods
    # ------------------------------------------------------------------

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background,
                         self._PD_MAX_SUPPORTED_DEPTH, "tree_path_dependent", False)

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background,
                         self._BG_MAX_SUPPORTED_DEPTH, "tree_path_dependent", True)

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background,
                         self._BG_MAX_SUPPORTED_DEPTH, "interventional", False)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background,
                         self._BG_MAX_SUPPORTED_DEPTH, "interventional", True)

    # ------------------------------------------------------------------
    # Complexity formulas
    # ------------------------------------------------------------------

    def complexity_formula_path_dependent_shap(self, params: TreeParameters) -> Optional[float]:
        return params.n * params.T * params.D + params.T * params.L * (2 ** params.D) * (params.D ** 2)

    def complexity_formula_path_dependent_interactions(self, params: TreeParameters) -> Optional[float]:
        return params.n * params.T * params.D + params.T * params.L * (2 ** params.D) * (params.D ** 3)

    def complexity_formula_background_shap(self, params: TreeParameters) -> Optional[float]:
        return (
            params.m * params.T * params.L
            + params.n * params.T * params.D
            + params.T * params.L * (2 ** params.D) * (params.D ** 2)
        )

    def complexity_formula_background_shap_interactions(self, params: TreeParameters) -> Optional[float]:
        return (
            params.m * params.T * params.L
            + params.n * params.T * params.D
            + params.T * params.L * (2 ** params.D) * (params.D ** 3)
        )

