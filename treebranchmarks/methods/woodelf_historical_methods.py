"""
Historical Woodelf algorithm implementations.

_WoodelfAlgorithmApproach
    Shared base for ECAI and AAAI approaches.
    memory_crash when D > MAX_SUPPORTED_DEPTH.

WoodelfECAIApproach
    ECAI WDNF-based algorithm. MAX_SUPPORTED_DEPTH=10.

WoodelfAAAIApproach
    AAAI cube-based algorithm. MAX_SUPPORTED_DEPTH=10.

WoodelfHDApproach
    woodelf_for_high_depth. Per-task depth limits:
      SHAP tasks (PD + BG):     extrapolate when D > 18, BG crashes at D >= 20
      IV tasks (PD + BG):       extrapolate when D > 15, BG crashes at D >= 18
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import WOODELF_ECAI, WOODELF_AAAI, WOODELF_HD
from treebranchmarks.tree_algs.woodelf_ECAI import (
    calculate_path_dependent_shap as ecai_calculate_path_dependent_shap,
    calculate_background_shap as ecai_calculate_background_shap,
    ShapleyValues as ECAIShapleyValues,
    ShapleyInteractionValues as ECAIShapleyInteractionValues,
)
from treebranchmarks.tree_algs.woodelf_AAAI import (
    calculate_path_dependent_shap as aaai_calculate_path_dependent_shap,
    calculate_background_shap as aaai_calculate_background_shap,
    ShapleyValues as AAAIShapleyValues,
    ShapleyInteractionValues as AAAIShapleyInteractionValues,
)
from woodelf.high_depth_woodelf import woodelf_for_high_depth
from woodelf.cube_metric import (
    ShapleyValues as HDShapleyValues,
    ShapleyInteractionValues as HDShapleyInteractionValues,
)
from woodelf.parse_models import load_decision_tree_ensemble_model
from woodelf.decision_trees_ensemble import DecisionTreesEnsemble


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _WoodelfAlgorithmApproach(Approach):
    """
    Base for ECAI, AAAI, and HD Approach classes.

    Implements all 4 task methods via _run_pd / _run_bg helpers.
    Subclasses declare four class attributes:
      _calculate_pd       — staticmethod wrapping calculate_path_dependent_shap
      _calculate_bg       — staticmethod wrapping calculate_background_shap
      _shap_values_cls    — the ShapleyValues class to instantiate
      _interaction_values_cls — the ShapleyInteractionValues class
    """

    MAX_SUPPORTED_DEPTH: int = 10

    # Set as staticmethod(fn) in subclasses to avoid Python's descriptor binding
    _calculate_pd: staticmethod  # calculate_path_dependent_shap
    _calculate_bg: staticmethod  # calculate_background_shap
    _shap_values_cls: type       # ShapleyValues
    _interaction_values_cls: type  # ShapleyInteractionValues

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_pd(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        metric,
    ) -> ApproachOutput:
        if trained_model.params.D > self.MAX_SUPPORTED_DEPTH:
            return ApproachOutput(elapsed_s=0.0, memory_crash=True)
        t0 = time.perf_counter()
        self._calculate_pd(trained_model.raw_model, X_explain, metric)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def _run_bg(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        metric,
    ) -> ApproachOutput:
        if trained_model.params.D > self.MAX_SUPPORTED_DEPTH:
            return ApproachOutput(elapsed_s=0.0, memory_crash=True)
        if X_background is None:
            raise ValueError(f"{self.name} requires X_background.")
        t0 = time.perf_counter()
        self._calculate_bg(trained_model.raw_model, X_explain, X_background, metric)
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
        return self._run_pd(trained_model, X_explain, self._shap_values_cls())

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run_pd(trained_model, X_explain, self._interaction_values_cls())

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run_bg(trained_model, X_explain, X_background, self._shap_values_cls())

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run_bg(trained_model, X_explain, X_background, self._interaction_values_cls())


# ---------------------------------------------------------------------------
# WoodelfECAIApproach
# ---------------------------------------------------------------------------

class WoodelfECAIApproach(_WoodelfAlgorithmApproach):
    """Woodelf ECAI algorithm (WDNF-based). Supports all 4 task types."""

    name = "Woodelf ECAI"
    method = WOODELF_ECAI
    description = "ECAI WDNF-based algorithm (calculate_path_dependent_shap / calculate_background_shap)."

    MAX_SUPPORTED_DEPTH = 10

    _calculate_pd           = staticmethod(ecai_calculate_path_dependent_shap)
    _calculate_bg           = staticmethod(ecai_calculate_background_shap)
    _shap_values_cls        = ECAIShapleyValues
    _interaction_values_cls = ECAIShapleyInteractionValues


# ---------------------------------------------------------------------------
# WoodelfAAAIApproach
# ---------------------------------------------------------------------------

class WoodelfAAAIApproach(_WoodelfAlgorithmApproach):
    """Woodelf AAAI algorithm (cube-based). Supports all 4 task types."""

    name = "Woodelf AAAI"
    method = WOODELF_AAAI
    description = "AAAI cube-based algorithm (calculate_path_dependent_shap / calculate_background_shap)."

    MAX_SUPPORTED_DEPTH = 10

    _calculate_pd           = staticmethod(aaai_calculate_path_dependent_shap)
    _calculate_bg           = staticmethod(aaai_calculate_background_shap)
    _shap_values_cls        = AAAIShapleyValues
    _interaction_values_cls = AAAIShapleyInteractionValues


# ---------------------------------------------------------------------------
# WoodelfHDApproach
# ---------------------------------------------------------------------------

class WoodelfHDApproach(Approach):
    """
    woodelf_for_high_depth implementation covering all 4 task types.

    Path-dependent tasks: background_data=None.
    Background (interventional) tasks: background_data=X_background.
    Metrics: ShapleyValues / ShapleyInteractionValues from woodelf.cube_metric.

    SHAP tasks (PD + BG):  extrapolate when D > 18, BG crashes at D >= 20.
    IV tasks (PD + BG):    extrapolate when D > 15, BG crashes at D >= 18.
    """

    name = "WoodelfHD"
    method = WOODELF_HD
    description = "woodelf_for_high_depth implementation (woodelf.cube_metric metrics)."

    _SHAP_TREE_LIMIT_DEPTH = 18
    _IV_TREE_LIMIT_DEPTH   = 15
    _BG_SHAP_CRASH_DEPTH   = 20   # background_shap: crash when D >= this
    _IV_CRASH_DEPTH        = 18   # PD/BG interactions: crash when D >= this

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _run_hd(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        metric,
        tree_limit_depth: int,
        memory_crash_depth: Optional[int],
    ) -> ApproachOutput:
        if memory_crash_depth is not None and trained_model.params.D >= memory_crash_depth:
            return ApproachOutput(elapsed_s=0.0, memory_crash=True)

        T = trained_model.params.T

        if trained_model.params.D > tree_limit_depth:
            model_obj = load_decision_tree_ensemble_model(
                trained_model.raw_model, list(X_explain.columns)
            )
            single_tree = DecisionTreesEnsemble([model_obj.trees[0]])
            t0 = time.perf_counter()
            woodelf_for_high_depth(single_tree, X_explain, X_background, metric, model_was_loaded=True)
            elapsed = time.perf_counter() - t0
            return ApproachOutput(
                elapsed_s=elapsed * T,
                is_estimated=True,
                estimation_description=(
                    f"D={trained_model.params.D} > {tree_limit_depth}: "
                    f"ran with tree_limit=1, extrapolated ×{T} trees"
                ),
            )

        t0 = time.perf_counter()
        woodelf_for_high_depth(trained_model.raw_model, X_explain, X_background, metric)
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
        return self._run_hd(trained_model, X_explain, None, HDShapleyValues(), self._SHAP_TREE_LIMIT_DEPTH, None)

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run_hd(trained_model, X_explain, None, HDShapleyInteractionValues(), self._IV_TREE_LIMIT_DEPTH, self._IV_CRASH_DEPTH)

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run_hd(trained_model, X_explain, X_background, HDShapleyValues(), self._SHAP_TREE_LIMIT_DEPTH, self._BG_SHAP_CRASH_DEPTH)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run_hd(trained_model, X_explain, X_background, HDShapleyInteractionValues(), self._IV_TREE_LIMIT_DEPTH, self._IV_CRASH_DEPTH)
