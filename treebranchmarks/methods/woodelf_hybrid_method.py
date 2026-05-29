"""
Hybrid Woodelf approach implementations.

Base class and Sparse variants call woodelf_sparse (always sparse, all depths).
HybridWoodelfAutoApproach calls hybrid_woodelf, which auto-selects sparse vs
woodelf_for_high_depth based on depth thresholds — this function lives in the
woodelf_explainer "feature/hybrid_woodelf" branch.

Requires the local woodelf branch (woodelf_explainer) installed in editable mode:
    pip install -e <path-to-woodelf_explainer>
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from woodelf.core.cube_metric import ShapleyValues, ShapleyInteractionValues
from woodelf.woodelf_sparse import hybrid_woodelf, woodelf_sparse

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import (
    WOODELF_HYBRID_SPARSE,
    WOODELF_HYBRID_SPARSE_NO_FAST_MN,
    WOODELF_HYBRID_SPARSE_NO_NEIGHBOR_TRICK,
    WOODELF_HYBRID_AUTO,
)


class _HybridWoodelfBaseApproach(Approach):
    """
    Base for all hybrid_woodelf variants. Calls woodelf_sparse directly.

    Subclasses may set `_use_faster_mn_p2s` or override `_run` entirely.
    """

    _use_faster_mn_p2s: bool = True
    _use_neighbor_leaf_trick: bool = True

    def _run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        metric,
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        woodelf_sparse(
            trained_model.raw_model,
            X_explain,
            X_background,
            metric,
            use_faster_mn_p2s=self._use_faster_mn_p2s,
            use_neighbor_leaf_trick=self._use_neighbor_leaf_trick,
        )
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, None, ShapleyValues())

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, None, ShapleyInteractionValues())

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background, ShapleyValues())

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background, ShapleyInteractionValues())


class HybridWoodelfSparseApproach(_HybridWoodelfBaseApproach):
    """woodelf_sparse at all depths (always uses MN/LTS sparse path)."""

    name = "HybridWoodelf (Sparse)"
    method = WOODELF_HYBRID_SPARSE
    description = "woodelf_sparse: always uses sparse (MN/LTS) path regardless of depth."
    _use_faster_mn_p2s = True


class HybridWoodelfSparseNoNeighborTrickApproach(_HybridWoodelfBaseApproach):
    """woodelf_sparse with use_neighbor_leaf_trick=False."""

    name = "HybridWoodelf (Sparse, no NLT)"
    method = WOODELF_HYBRID_SPARSE_NO_NEIGHBOR_TRICK
    description = "woodelf_sparse with use_neighbor_leaf_trick=False."
    _use_neighbor_leaf_trick = False


class HybridWoodelfSparseNoFastMNApproach(_HybridWoodelfBaseApproach):
    """Like HybridWoodelfSparseApproach but uses MNBackgroundPathToSVectors instead of the faster variant."""

    name = "HybridWoodelf (Sparse, slow MN)"
    method = WOODELF_HYBRID_SPARSE_NO_FAST_MN
    description = "woodelf_sparse with use_faster_mn_p2s=False."
    _use_faster_mn_p2s = False


class HybridWoodelfAutoApproach(_HybridWoodelfBaseApproach):
    """
    hybrid_woodelf: auto-selects sparse vs woodelf_for_high_depth based on depth thresholds.

    Note: hybrid_woodelf lives in the woodelf_explainer "feature/hybrid_woodelf" branch.
    """

    name = "HybridWoodelf (Auto)"
    method = WOODELF_HYBRID_AUTO
    description = (
        "hybrid_woodelf (feature/hybrid_woodelf branch): auto-selects sparse vs HD strategy "
        "based on depth thresholds."
    )

    def _run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        metric,
    ) -> ApproachOutput:
        t0 = time.perf_counter()
        hybrid_woodelf(
            trained_model.raw_model,
            X_explain,
            X_background,
            metric,
            use_faster_mn_p2s=self._use_faster_mn_p2s,
        )
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)
