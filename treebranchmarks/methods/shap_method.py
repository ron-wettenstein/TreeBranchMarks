"""
SHAP library (shap.TreeExplainer) implementation.

Supported tasks
---------------
- path_dependent_shap        : TreeExplainer(feature_perturbation="tree_path_dependent")
- path_dependent_interactions: shap_interaction_values, tree_path_dependent
- background_shap            : TreeExplainer(feature_perturbation="interventional")
- background_shap_interactions: not supported by the shap library
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import shap

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import SHAP


class SHAPApproach(Approach):
    """SHAP library implementation covering all supported task types."""

    name = "SHAP"
    method = SHAP
    description = "Reference implementation from the shap library (shap.TreeExplainer)."

    _BG_DEPTH_THRESHOLD = 18
    _BG_SAMPLE_LIMIT  = 10_000

    def __init__(self, bg_shap_limit: int = 100) -> None:
        """
        Parameters
        ----------
        bg_shap_limit : int
            Maximum number of background rows passed to shap.TreeExplainer.
            Elapsed time is scaled by m / bg_shap_limit when m > bg_shap_limit.
            Default: 100.
        """
        self._BG_SHAP_LIMIT = bg_shap_limit

    # ---------------------------------------------------------------------------
    # Task methods
    # ---------------------------------------------------------------------------

    def path_dependent_shap(
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
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def path_dependent_interactions(
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
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        if X_background is None:
            raise ValueError("SHAPApproach.background_shap requires X_background.")

        n = len(X_explain)
        m = len(X_background)
        n_subsample = trained_model.params.D > self._BG_DEPTH_THRESHOLD and n > self._BG_SAMPLE_LIMIT
        X_run = X_explain.iloc[:self._BG_SAMPLE_LIMIT] if n_subsample else X_explain

        if m > self._BG_SHAP_LIMIT:
            X_background = X_background.sample(self._BG_SHAP_LIMIT, random_state=42)

        explainer = shap.TreeExplainer(
            trained_model.raw_model,
            data=X_background,
            feature_perturbation="interventional",
        )
        t0 = time.perf_counter()
        explainer.shap_values(X_run, check_additivity=False)
        elapsed = time.perf_counter() - t0

        n_scale = n / self._BG_SAMPLE_LIMIT if n_subsample else 1.0
        m_scale = m / self._BG_SHAP_LIMIT if m > self._BG_SHAP_LIMIT else 1.0
        total_scale = n_scale * m_scale

        if total_scale != 1.0:
            desc_parts = []
            if n_subsample:
                desc_parts.append(
                    f"D={trained_model.params.D} > {self._BG_DEPTH_THRESHOLD}: "
                    f"ran with {self._BG_SAMPLE_LIMIT} samples, extrapolated ×{n_scale:.1f} to n={n}\n"
                )
            if m > self._BG_SHAP_LIMIT:
                desc_parts.append(
                    f"ran with m={self._BG_SHAP_LIMIT} background rows \n"
                    f"(shap internal limit), extrapolated ×{m_scale:.1f} to m={m}\n"
                )
            return ApproachOutput(
                elapsed_s=elapsed * total_scale,
                is_estimated=True,
                estimation_description="; ".join(desc_parts),
            )
        return ApproachOutput(elapsed_s=elapsed)

    # background_shap_interactions: not supported (inherits not_supported default)
