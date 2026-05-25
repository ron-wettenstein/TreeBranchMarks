"""
Hybrid Woodelf approach implementations.

All variants call woodelf.hybrid_woodelf.hybrid_woodelf with different kwargs:
  - HybridWoodelfSparseApproach:         use_sparse_approaches=True
  - HybridWoodelfSparseNoFastMNApproach: use_sparse_approaches=True, use_faster_mn_p2s=False
  - HybridWoodelfAutoApproach:           use_sparse_approaches=False (auto per-leaf selection)

Requires the local woodelf branch (woodelf_explainer) installed in editable mode:
    pip install -e <path-to-woodelf_explainer>
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from woodelf.core.cube_metric import ShapleyValues, ShapleyInteractionValues
from woodelf.hybrid_woodelf import hybrid_woodelf

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import (
    WOODELF_HYBRID_SPARSE,
    WOODELF_HYBRID_SPARSE_NO_FAST_MN,
    WOODELF_HYBRID_AUTO,
)


class _HybridWoodelfBaseApproach(Approach):
    """
    Base for all hybrid_woodelf variants.

    Subclasses set `_use_sparse_approaches` and `_use_faster_mn_p2s` as class attributes.
    Note: `_use_faster_mn_p2s` has no effect for path-dependent tasks (m=0) — the two
    sparse variants only differ in background tasks.
    """

    _use_sparse_approaches: bool = False
    _use_faster_mn_p2s: bool = True

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
            use_sparse_approaches=self._use_sparse_approaches,
            use_faster_mn_p2s=self._use_faster_mn_p2s,
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
    """hybrid_woodelf always using sparse (MN/LTS) paths. Best for high-depth trees."""

    name = "HybridWoodelf (Sparse)"
    method = WOODELF_HYBRID_SPARSE
    description = "hybrid_woodelf with use_sparse_approaches=True."
    _use_sparse_approaches = True
    _use_faster_mn_p2s = True


class HybridWoodelfSparseNoFastMNApproach(_HybridWoodelfBaseApproach):
    """Like HybridWoodelfSparseApproach but uses MNBackgroundPathToSVectors instead of the faster variant."""

    name = "HybridWoodelf (Sparse, slow MN)"
    method = WOODELF_HYBRID_SPARSE_NO_FAST_MN
    description = "hybrid_woodelf with use_sparse_approaches=True, use_faster_mn_p2s=False."
    _use_sparse_approaches = True
    _use_faster_mn_p2s = False


class HybridWoodelfAutoApproach(_HybridWoodelfBaseApproach):
    """hybrid_woodelf auto-selecting sparse vs dense strategy per leaf based on depth and data size."""

    name = "HybridWoodelf (Auto)"
    method = WOODELF_HYBRID_AUTO
    description = "hybrid_woodelf with use_sparse_approaches=False (auto per-leaf strategy selection)."
    _use_sparse_approaches = False
    _use_faster_mn_p2s = True
