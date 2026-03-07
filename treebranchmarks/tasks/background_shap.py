"""
Background (Interventional) SHAP.

Algorithm
---------
Uses an explicit background dataset to estimate the expected value of each
feature, independent of the other features.  This corresponds to the
"interventional" feature perturbation in the SHAP library.

Theoretical complexity: O(T * L * n * m)
  - T trees, each with L leaves
  - n rows to explain, m background rows
  - For each (explain row, background row) pair the algorithm marginalises
    features not in the coalition by substituting background values

Reference implementation: shap.TreeExplainer with
    feature_perturbation="interventional"
    and a background dataset passed to the constructor.
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

class BackgroundSHAPApproach(Approach):
    """
    SHAP TreeExplainer with feature_perturbation="interventional".

    Requires a background dataset.  X_background must have at least 1 row.

    Complexity: O(T * L * n * m)
    """

    name = "shap Package Background SHAP"
    method = SHAP
    description = (
        "O(nmTLD)."
    )

    # shap.TreeExplainer internally uses only the first 100 background rows.
    # When m > 100 we cap the background at 100, measure the real time, and
    # scale it up by m/100 so the returned elapsed_s reflects the full cost.
    _SHAP_BG_LIMIT = 100

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        import shap

        if X_background is None:
            raise ValueError(
                "BackgroundSHAPApproach requires a background dataset (X_background is None)."
            )

        m = len(X_background)
        if m > self._SHAP_BG_LIMIT:
            X_background = X_background.sample(self._SHAP_BG_LIMIT, random_state=42)
        explainer = shap.TreeExplainer(
            trained_model.raw_model,
            data=X_background,
            feature_perturbation="interventional",
        )
        t0 = time.perf_counter()
        explainer.shap_values(X_explain, check_additivity=False)
        elapsed = time.perf_counter() - t0

        if m > self._SHAP_BG_LIMIT:
            scale = m / self._SHAP_BG_LIMIT
            return ApproachOutput(elapsed_s=elapsed * scale, is_estimated=True)
        return ApproachOutput(elapsed_s=elapsed, is_estimated=False)

    def complexity_formula(self, params: TreeParameters) -> Optional[float]:
        # O(T * L * n * m)
        return params.T * params.L * params.n * params.m


class WoodelfBackgroundSHAPApproach(Approach):
    """
    SHAP TreeExplainer with feature_perturbation="interventional".

    Requires a background dataset.  X_background must have at least 1 row.

    Complexity: O(T * L * n * m)
    """

    name = "Woodelf Background SHAP"
    method = WOODELF
    description = (
        "Complexity: O(mTL + nTLD + TL2**D * D²)."
    )

    _SHAP_BG_LIMIT = 100

    MAX_SUPPORTED_DEPTH = 21

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:

        if trained_model.params.D > self.MAX_SUPPORTED_DEPTH:
            return ApproachOutput(elapsed_s=0.0, not_supported=True)

        if X_background is None:
            raise ValueError(
                "WoodelfBackgroundSHAPApproach requires a background dataset (X_background is None)."
            )

        explainer = WoodelfExplainer(
            trained_model.raw_model,
            data=X_background,
            feature_perturbation="interventional",
        )
        t0 = time.perf_counter()
        explainer.shap_values(X_explain)
        elapsed = time.perf_counter() - t0

        return ApproachOutput(elapsed_s=elapsed, is_estimated=False)

    def complexity_formula(self, params: TreeParameters) -> Optional[float]:
        # O(T * L * n * m)
        return params.m * params.T * params.L + params.n * params.T * params.D  + params.T * params.L * (2 ** params.D)* (params.D ** 2)


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def BackgroundSHAPTask(
    methods: list | None = None,
    extra_approaches: list[Approach] | None = None,
    n_repeats: int = 3,
    cache_root: Path = Path("cache"),
) -> Task:
    """
    Convenience factory that returns a Task preconfigured for background SHAP.

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
        SHAP:    BackgroundSHAPApproach,
        WOODELF: WoodelfBackgroundSHAPApproach,
    }
    if methods is None:
        methods = [SHAP, WOODELF]

    approaches: list[Approach] = [_registry[m]() for m in methods if m in _registry]
    if extra_approaches:
        approaches.extend(extra_approaches)

    return Task(
        name="Background SHAP",
        approaches=approaches,
        n_repeats=n_repeats,
        cache_root=cache_root,
    )
