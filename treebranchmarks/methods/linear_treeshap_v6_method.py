"""
LinearTreeSHAPV6Approach — path-dependent SHAP using linear_treeshap_v6_woodelf.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import LINEAR_TREESHAP_V6
from treebranchmarks.tree_algs.linear_treeshap_v6 import linear_treeshap_v6_woodelf
from woodelf.parse_models import load_decision_tree_ensemble_model


class LinearTreeSHAPV6Approach(Approach):
    """Path-dependent SHAP via the Linear TreeSHAP V6 algorithm (telescoping + quadrature)."""

    name = "Linear TreeSHAP V6"
    method = LINEAR_TREESHAP_V6
    description = "Path-dependent SHAP using linear_treeshap_v6_woodelf (telescoping + Gauss-Legendre quadrature)."

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        ensemble = load_decision_tree_ensemble_model(
            trained_model.raw_model, list(X_explain.columns)
        )
        t0 = time.perf_counter()
        linear_treeshap_v6_woodelf(ensemble, X_explain)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)
