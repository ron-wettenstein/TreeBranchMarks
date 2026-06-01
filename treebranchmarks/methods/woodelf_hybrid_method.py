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
from woodelf.core.cube_metric import ShapleyValues, ShapleyInteractionValues, CubeMetric
from woodelf.core.path_to_s_vectors.mn_background_p2s import (
    MNBackgroundPathToSVectors,
    MNBackgroundFasterPathToSVectors,
)
from woodelf.core.utils import bits_matrix
from woodelf.woodelf_sparse import hybrid_woodelf, woodelf_sparse

import numpy as np
import scipy


from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import (
    WOODELF_HYBRID_SPARSE,
    WOODELF_HYBRID_SPARSE_NO_FAST_MN,
    WOODELF_HYBRID_SPARSE_NO_NEIGHBOR_TRICK,
    WOODELF_HYBRID_SPARSE_DIRECT_MN,
    WOODELF_HYBRID_AUTO,
)


class _HybridWoodelfBaseApproach(Approach):
    """
    Base for all hybrid_woodelf variants. Calls woodelf_sparse directly.

    Subclasses may set `_mn_p2s_class` or `_use_neighbor_leaf_trick`, or override `_run` entirely.
    """

    _mn_p2s_class = None  # None → MNBackgroundFasterPathToSVectors (default)
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
            use_neighbor_leaf_trick=self._use_neighbor_leaf_trick,
            mn_p2s_class=self._mn_p2s_class,
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
    description = "woodelf_sparse with mn_p2s_class=MNBackgroundPathToSVectors."
    _mn_p2s_class = MNBackgroundPathToSVectors


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
        )
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)



class MNBackgroundShapleyDirectPathToSVectors(MNBackgroundFasterPathToSVectors):
    """
    Variant of MNBackgroundPathToSVectors for ShapleyValues that computes pos_weighted and
    neg_weighted directly via the closed-form Shapley formula, avoiding the precomputed
    (D+1)x(D+1) contribution tables entirely.

    For a cube with p positive and n negative literals:
        pos_weighted[c,b] = 1 / (p · C(p+n, p))
        neg_weighted[c,b] = -1 / (n · C(p+n, n))

    Since C(p+n, p) == C(p+n, n), only one binomial evaluation per (c,b) pair is needed.
    """

    def __init__(self, metric: CubeMetric, max_depth: int, GPU: bool = False):
        super().__init__(metric, max_depth, GPU)
        assert isinstance(metric, ShapleyValues), (
            f"MNBackgroundShapleyDirectPathToSVectors requires ShapleyValues, got {type(metric).__name__}."
        )

    def _compute_s_batch(
        self, consumer_batch: np.ndarray, unique_b: np.ndarray,
        b_freqs: np.ndarray, D: int, w: float
    ) -> np.ndarray:
        diff     = consumer_batch[:, None] ^ unique_b[None, :]       # (batch, U_b)
        positive = consumer_batch[:, None] & diff                     # s_plus bits per pair
        negative = unique_b[None, :]       & diff                     # s_minus bits per pair
        valid = (consumer_batch[:, None] | unique_b[None, :]) == ((1 << D) - 1)  # (batch, U_b)

        p = np.bitwise_count(positive)                                # (batch, U_b)
        n = np.bitwise_count(negative)                                # (batch, U_b)

        binom_pn_p = scipy.special.comb(p + n, p, exact=False)                   # C(p+n,p) == C(p+n,n)
        # When p=0 pos_bits=0, so pos_contribs is multiplied away; clamp to avoid div-by-zero.
        pos_weighted = (1.0 / (np.maximum(p, 1) * binom_pn_p)) * b_freqs * valid  # (batch, U_b)
        neg_weighted = (-1.0 / (np.maximum(n, 1) * binom_pn_p)) * b_freqs * valid  # (batch, U_b)

        c_bits = bits_matrix(consumer_batch, D).astype(np.float64)    # (D, batch)
        b_bits = bits_matrix(unique_b, D).astype(np.float64)          # (D, U_b)

        BPW = b_bits @ pos_weighted.T                                  # (D, batch)
        BNW = b_bits @ neg_weighted.T                                  # (D, batch)
        total_pos = pos_weighted.sum(axis=1)                           # (batch,)
        return (BNW + c_bits * (total_pos - BPW - BNW)) * w           # (D, batch)
    
class HybridWoodelfSparseDirectMNApproach(_HybridWoodelfBaseApproach):
    """woodelf_sparse with MNBackgroundShapleyDirectPathToSVectors and neighbor leaf trick enabled."""

    name = "HybridWoodelf (Sparse, direct MN)"
    method = WOODELF_HYBRID_SPARSE_DIRECT_MN
    description = "woodelf_sparse with mn_p2s_class=MNBackgroundShapleyDirectPathToSVectors."
    _mn_p2s_class = MNBackgroundShapleyDirectPathToSVectors