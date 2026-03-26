"""
PLTreeSHAP + FastTreeSHAP combined implementation.

Task routing
------------
- path_dependent_shap        : fasttreeshap.TreeExplainer (algorithm="v2")
- path_dependent_interactions: fasttreeshap.TreeExplainer (algorithm="v1")
- background_shap            : pltreeshap.PLTreeExplainer
- background_shap_interactions: pltreeshap.PLTreeExplainer
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import PLTREESHAP_FASTTREESHAP


class PLTreeSHAPFastTreeSHAPApproach(Approach):
    """PLTreeSHAP for background tasks; FastTreeSHAP for path-dependent tasks."""

    name = "PLTreeSHAP + FastTreeSHAP"
    method = PLTREESHAP_FASTTREESHAP
    description = (
        "PLTreeSHAP (pltreeshap.PLTreeExplainer) for background SHAP tasks; "
        "FastTreeSHAP (fasttreeshap.TreeExplainer) for path-dependent SHAP tasks."
    )

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        import fasttreeshap
        t0 = time.perf_counter()
        explainer = fasttreeshap.TreeExplainer(
            trained_model.raw_model, algorithm="v2", n_jobs=1
        )
        explainer(X_explain, check_additivity=False).values
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        import fasttreeshap
        t0 = time.perf_counter()
        explainer = fasttreeshap.TreeExplainer(
            trained_model.raw_model, algorithm="v1", n_jobs=1
        )
        explainer(X_explain, check_additivity=False, interactions=True).values
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        from pltreeshap import PLTreeExplainer
        if X_background is None:
            raise ValueError("PLTreeSHAPFastTreeSHAPApproach.background_shap requires X_background.")
        t0 = time.perf_counter()
        explainer = PLTreeExplainer(trained_model.raw_model)
        explainer.aggregate(X_background)
        explainer.shap_values(X_explain)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        from pltreeshap import PLTreeExplainer
        if X_background is None:
            raise ValueError("PLTreeSHAPFastTreeSHAPApproach.background_shap_interactions requires X_background.")
        t0 = time.perf_counter()
        explainer = PLTreeExplainer(trained_model.raw_model)
        explainer.aggregate(X_background)
        explainer.shap_interaction_values(X_explain)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)
