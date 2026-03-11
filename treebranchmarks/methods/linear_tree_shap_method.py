"""
Vectorized Linear TreeSHAP implementations.

All variants only support path_dependent_shap.  Other tasks return not_supported.

Classes
-------
VectorizedLinearTreeSHAPApproach             — default params, neighbor-leaf trick ON
VectorizedLinearTreeSHAPSimpleApproach       — Simple p2m, no NLT
VectorizedLinearTreeSHAPSimpleNLTApproach    — Simple p2m, NLT ON
VectorizedLinearTreeSHAPImprovedApproach     — Improved p2m, no NLT
VectorizedLinearTreeSHAPImprovedNLTApproach  — Improved p2m, NLT ON
VectorizedLinearTreeSHAPDefaultApproach      — default p2m, no NLT
VectorizedLinearTreeSHAPDefaultNLTApproach   — default p2m, NLT ON
VectorizedLinearTreeSHAPRecursiveNLTApproach — Recursive (SimpleNeighborTrickAbstract + improved_magic), NLT ON
"""

from __future__ import annotations

import time
from typing import Optional, Type

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
import numpy as np

from treebranchmarks.methods.builtin import (
    VECTORIZED_LINEAR_TREE_SHAP,
    WOODELF_VEC_SIMPLE,
    WOODELF_VEC_SIMPLE_NLT,
    WOODELF_VEC_IMPROVED,
    WOODELF_VEC_IMPROVED_NLT,
    WOODELF_VEC_DEFAULT,
    WOODELF_VEC_DEFAULT_NLT,
    WOODELF_VEC_RECURSIVE_NLT,
)
from woodelf.lts_vectorized import (
    vectorized_linear_tree_shap,
    LinearTreeShapPathToMatrices,
    LinearTreeShapPathToMatricesImproved,
    LinearTreeShapPathToMatricesSimple,
)
from woodelf.lts_polynomial_multiplication import (
    improved_linear_tree_shap_magic,
    improved_linear_tree_shap_magic_for_neighbors,
    linear_tree_shap_magic_for_banzhaf,
)


class _LinearTreeShapPathToMatricesRecursiveNLT(LinearTreeShapPathToMatrices):
    def get_s_matrix(self, covers: np.array, consumer_patterns: np.array, w: float, w_neighbor: Optional[float] = None):
        start_time = time.time()
        if self.is_shapley:
            f_w = self.f_ws[len(covers)]
            if w_neighbor is None:
                s_matrix = improved_linear_tree_shap_magic(covers, consumer_patterns, f_w, w)
            else:
                s_matrix = improved_linear_tree_shap_magic_for_neighbors(covers, consumer_patterns, f_w, w, w_neighbor)
        else:
            if w_neighbor is not None:
                s_matrix_left = linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)
                covers_of_right = np.array(list(covers[:-1]) + [1 - covers[-1]])
                consumer_patterns_right = consumer_patterns.copy()
                consumer_patterns_right[consumer_patterns % 2 == 0] += 1
                consumer_patterns_right[consumer_patterns % 2 == 1] -= 1
                s_matrix_right = linear_tree_shap_magic_for_banzhaf(covers_of_right, consumer_patterns_right.astype(np.uint64), w_neighbor)
                s_matrix = s_matrix_left + s_matrix_right
            else:
                s_matrix = linear_tree_shap_magic_for_banzhaf(covers, consumer_patterns, w)
        self.computation_time += time.time() - start_time
        return s_matrix


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VectorizedLinearTreeSHAPBase(Approach):
    """
    Base for all vectorized_linear_tree_shap variants.

    Only path_dependent_shap is supported.

    Subclasses set:
      _use_neighbor_leaf_trick : bool
      _p2m_class               : type | None  (None → use library default)
    """

    _use_neighbor_leaf_trick: bool = False
    _p2m_class: Optional[Type] = None

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        kwargs: dict = {
            "is_shapley": True,
            "use_neighbor_leaf_trick": self._use_neighbor_leaf_trick,
        }
        if self._p2m_class is not None:
            kwargs["p2m_class"] = self._p2m_class

        t0 = time.perf_counter()
        vectorized_linear_tree_shap(trained_model.raw_model, X_explain, **kwargs)
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------

class VectorizedLinearTreeSHAPApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap with default p2m and neighbor-leaf trick ON."""

    name = "VectorizedLinearTreeSHAP"
    method = VECTORIZED_LINEAR_TREE_SHAP
    description = "vectorized_linear_tree_shap with default params and neighbor-leaf trick."
    _use_neighbor_leaf_trick = True
    _p2m_class = None


class VectorizedLinearTreeSHAPSimpleApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap with Simple p2m, no neighbor-leaf trick."""

    name = "Woodelf Vec Simple"
    method = WOODELF_VEC_SIMPLE
    description = "vectorized_linear_tree_shap(p2m=Simple, neighbor_leaf_trick=False)"
    _use_neighbor_leaf_trick = False
    _p2m_class = LinearTreeShapPathToMatricesSimple


class VectorizedLinearTreeSHAPSimpleNLTApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap with Simple p2m and neighbor-leaf trick."""

    name = "Woodelf Vec Simple + NLT"
    method = WOODELF_VEC_SIMPLE_NLT
    description = "vectorized_linear_tree_shap(p2m=Simple, neighbor_leaf_trick=True)"
    _use_neighbor_leaf_trick = True
    _p2m_class = LinearTreeShapPathToMatricesSimple


class VectorizedLinearTreeSHAPImprovedApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap with Improved p2m, no neighbor-leaf trick."""

    name = "Woodelf Vec Improved"
    method = WOODELF_VEC_IMPROVED
    description = "vectorized_linear_tree_shap(p2m=Improved, neighbor_leaf_trick=False)"
    _use_neighbor_leaf_trick = False
    _p2m_class = LinearTreeShapPathToMatricesImproved


class VectorizedLinearTreeSHAPImprovedNLTApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap with Improved p2m and neighbor-leaf trick."""

    name = "Woodelf Vec Improved + NLT"
    method = WOODELF_VEC_IMPROVED_NLT
    description = "vectorized_linear_tree_shap(p2m=Improved, neighbor_leaf_trick=True)"
    _use_neighbor_leaf_trick = True
    _p2m_class = LinearTreeShapPathToMatricesImproved


class VectorizedLinearTreeSHAPDefaultApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap with default p2m, no neighbor-leaf trick."""

    name = "Woodelf Vec Default"
    method = WOODELF_VEC_DEFAULT
    description = "vectorized_linear_tree_shap(neighbor_leaf_trick=False)"
    _use_neighbor_leaf_trick = False
    _p2m_class = None


class VectorizedLinearTreeSHAPDefaultNLTApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap with default p2m and neighbor-leaf trick."""

    name = "Woodelf Vec Default + NLT"
    method = WOODELF_VEC_DEFAULT_NLT
    description = "vectorized_linear_tree_shap(neighbor_leaf_trick=True)"
    _use_neighbor_leaf_trick = True
    _p2m_class = None


class VectorizedLinearTreeSHAPRecursiveNLTApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap with SimpleNeighborTrickAbstract + improved_magic and NLT."""

    name = "Woodelf Vec Recursive + NLT"
    method = WOODELF_VEC_RECURSIVE_NLT
    description = "vectorized_linear_tree_shap(p2m=RecursiveNLT, neighbor_leaf_trick=True)"
    _use_neighbor_leaf_trick = True
    _p2m_class = _LinearTreeShapPathToMatricesRecursiveNLT
