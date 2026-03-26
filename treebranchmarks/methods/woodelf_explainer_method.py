"""
WoodelfApproach — WoodelfExplainer covering all 4 task types.

- path_dependent_shap:          no limit
- path_dependent_interactions:  extrapolate when D > 15, crash when D >= 18
- background_shap:              extrapolate when D > 18, crash when D >= 20
- background_shap_interactions: extrapolate when D > 15, crash when D >= 18
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
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

    _BG_SHAP_TREE_LIMIT_DEPTH  = 18   # background_shap: extrapolate when D > this
    _BG_SHAP_CRASH_DEPTH       = 20   # background_shap: crash when D >= this
    _IV_TREE_LIMIT_DEPTH       = 15   # PD/BG interactions: extrapolate when D > this
    _IV_CRASH_DEPTH            = 18   # PD/BG interactions: crash when D >= this
    GPU                        = False

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        tree_limit_depth: Optional[int],
        memory_crash_depth: Optional[int],
        feature_perturbation: str,
        use_interactions: bool,
    ) -> ApproachOutput:
        D, T = trained_model.params.D, trained_model.params.T

        if memory_crash_depth is not None and D >= memory_crash_depth:
            return ApproachOutput(elapsed_s=0.0, memory_crash=True)

        if feature_perturbation == "interventional":
            if X_background is None:
                raise ValueError("WoodelfApproach interventional tasks require X_background.")
            explainer = WoodelfExplainer(
                trained_model.raw_model,
                data=X_background,
                feature_perturbation="interventional",
                GPU=self.GPU,
            )
        else:
            explainer = WoodelfExplainer(
                trained_model.raw_model,
                feature_perturbation="tree_path_dependent",
                GPU=self.GPU,
            )

        if tree_limit_depth is not None and D > tree_limit_depth:
            t0 = time.perf_counter()
            if use_interactions:
                explainer.shap_interaction_values(
                    X_explain, tree_limit=1, include_interaction_with_itself=False,
                    as_df=True, exclude_zero_contribution_features=True
                )
            else:
                explainer.shap_values(X_explain, tree_limit=1)
            elapsed = time.perf_counter() - t0
            return ApproachOutput(
                elapsed_s=elapsed * T,
                is_estimated=True,
                estimation_description=(
                    f"D={D} > {tree_limit_depth}: "
                    f"ran with tree_limit=1, extrapolated ×{T} trees"
                ),
            )

        t0 = time.perf_counter()
        if use_interactions:
            explainer.shap_interaction_values(
                X_explain, include_interaction_with_itself=False,
                as_df=True, exclude_zero_contribution_features=True
            )
        else:
            explainer.shap_values(X_explain)
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
                         None, None, "tree_path_dependent", False)

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background,
                         self._IV_TREE_LIMIT_DEPTH, self._IV_CRASH_DEPTH, "tree_path_dependent", True)

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background,
                         self._BG_SHAP_TREE_LIMIT_DEPTH, self._BG_SHAP_CRASH_DEPTH, "interventional", False)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background,
                         self._IV_TREE_LIMIT_DEPTH, self._IV_CRASH_DEPTH, "interventional", True)


# ---------------------------------------------------------------------------
# GPU variant
# ---------------------------------------------------------------------------

class WoodelfGPUApproach(WoodelfApproach):
    """WoodelfApproach with GPU=True (requires CuPy: pip install cupy)."""

    name = "Woodelf GPU"
    description = "Woodelf TreeExplainer implementation accelerated on GPU (CuPy required)."
    GPU = True
