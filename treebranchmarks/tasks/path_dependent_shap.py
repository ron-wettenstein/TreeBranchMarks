"""
Path-Dependent SHAP (TreeSHAP, interventional=False).

Algorithm
---------
Uses the decision path through the tree to compute SHAP values without a
background dataset.  The data distribution is estimated from the training
data that passes through each internal node.

Theoretical complexity: O(T * L^2 * n)
  - T trees, each with L leaves
  - For each of the n rows, the algorithm walks the tree and at each split
    considers which leaves are reachable from each ancestor

Reference implementation: shap.TreeExplainer with
    feature_perturbation="tree_path_dependent"
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Type

import numpy as np
import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.core.params import TreeParameters
from treebranchmarks.core.task import Task
from treebranchmarks.methods.builtin import (
    SHAP,
    WOODELF,
    WOODELF_VEC_SIMPLE,
    WOODELF_VEC_SIMPLE_NLT,
    WOODELF_VEC_IMPROVED,
    WOODELF_VEC_IMPROVED_NLT,
    WOODELF_VEC_DEFAULT,
    WOODELF_VEC_DEFAULT_NLT,
    VECTORIZED_LINEAR_TREE_SHAP
)
from woodelf import WoodelfExplainer
from woodelf.lts_vectorized import (
    vectorized_linear_tree_shap,
    LinearTreeShapPathToMatricesImproved,
    LinearTreeShapPathToMatricesSimple,
)
import shap


# ---------------------------------------------------------------------------
# Approach
# ---------------------------------------------------------------------------

class SHAPTreePathDependentApproach(Approach):
    """
    SHAP TreeExplainer with feature_perturbation="tree_path_dependent".

    This approach does not require a background dataset.  X_background is
    ignored.

    Complexity: O(T * L^2 * n)
    """

    name = "shap Package Path-Dependent SHAP"
    method = SHAP
    description = (
        "Complexity: O(nTLD²)."
    )

    _DEPTH_THRESHOLD = 25   # run on subsample when D >= this
    _SAMPLE_LIMIT    = 10_000

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        # n = len(X_explain)
        # if trained_model.params.D >= self._DEPTH_THRESHOLD and n > self._SAMPLE_LIMIT:
        #     X_sample = X_explain.iloc[: self._SAMPLE_LIMIT]
        #     explainer = shap.TreeExplainer(
        #         trained_model.raw_model,
        #         feature_perturbation="tree_path_dependent",
        #     )
        #     t0 = time.perf_counter()
        #     explainer.shap_values(X_sample, check_additivity=False)
        #     elapsed = time.perf_counter() - t0
        #     scale = n / self._SAMPLE_LIMIT
        #     return ApproachOutput(
        #         elapsed_s=elapsed * scale,
        #         is_estimated=True,
        #         estimation_description=(
        #             f"D={trained_model.params.D} ≥ {self._DEPTH_THRESHOLD}: "
        #             f"ran with {self._SAMPLE_LIMIT} samples, extrapolated ×{scale:.1f} to n={n}"
        #         ),
        #     )

        explainer = shap.TreeExplainer(
            trained_model.raw_model,
            feature_perturbation="tree_path_dependent",
        )
        t0 = time.perf_counter()
        explainer.shap_values(X_explain, check_additivity=False)
        elapsed = time.perf_counter() - t0

        return ApproachOutput(elapsed_s=elapsed)

    def complexity_formula(self, params: TreeParameters) -> Optional[float]:
        # O(T * L^2 * n)
        return params.T * (params.L ** 2) * params.n


class WoodelfSHAPTreePathDependentApproach(Approach):
    """
    Woodelf WoodelfExplainer with feature_perturbation="tree_path_dependent".

    This approach does not require a background dataset.  X_background is
    ignored.
    """

    name = "Woodelf Path-Dependent SHAP"
    method = WOODELF
    description = (
        "Complexity: O(nTLD + TL2**D * D²)."
    )

    MAX_SUPPORTED_DEPTH = 21
    _TREE_LIMIT_DEPTH   = 15   # run with tree_limit=1 when D >= this

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        if trained_model.params.D > self.MAX_SUPPORTED_DEPTH:
            return ApproachOutput(elapsed_s=0.0, memory_crash=True)

        T = trained_model.params.T

        if trained_model.params.D >= self._TREE_LIMIT_DEPTH:
            explainer = WoodelfExplainer(
                trained_model.raw_model,
                feature_perturbation="tree_path_dependent",
            )
            t0 = time.perf_counter()
            explainer.shap_values(X_explain, tree_limit=1)
            elapsed = time.perf_counter() - t0
            return ApproachOutput(
                elapsed_s=elapsed * T,
                is_estimated=True,
                estimation_description=(
                    f"D={trained_model.params.D} ≥ {self._TREE_LIMIT_DEPTH}: "
                    f"ran with tree_limit=1, extrapolated ×{T} trees"
                ),
            )

        explainer = WoodelfExplainer(
            trained_model.raw_model,
            feature_perturbation="tree_path_dependent",
        )
        t0 = time.perf_counter()
        explainer.shap_values(X_explain)
        elapsed = time.perf_counter() - t0

        return ApproachOutput(elapsed_s=elapsed)

    def complexity_formula(self, params: TreeParameters) -> Optional[float]:
        return params.n * params.T * params.D + params.T * params.L * (2 ** params.D) * (params.D ** 2)


# ---------------------------------------------------------------------------
# Vectorized Linear TreeSHAP approaches
# ---------------------------------------------------------------------------

class VectorizedLinearTreeSHAPBase(Approach):
    """
    Base class for vectorized_linear_tree_shap variants.

    Subclasses set:
      _use_neighbor_leaf_trick : bool
      _p2m_class               : type | None  (None → omit, use library default)
    """

    _use_neighbor_leaf_trick: bool = False
    _p2m_class: Optional[Type] = None   # None means don't pass p2m_class

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        kwargs = {
            "is_shapley": True,
            "use_neighbor_leaf_trick": self._use_neighbor_leaf_trick,
        }
        if self._p2m_class is not None:
            kwargs["p2m_class"] = self._p2m_class

        t0 = time.perf_counter()
        vectorized_linear_tree_shap(trained_model.raw_model, X_explain, **kwargs)
        elapsed = time.perf_counter() - t0

        return ApproachOutput(elapsed_s=elapsed)


class VectorizedLinearTreeSHAP(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap, Simple p2m, no neighbor-leaf trick."""

    name = "VectorizedLinearTreeSHAP"
    method = VECTORIZED_LINEAR_TREE_SHAP
    description = "vectorized_linear_tree_shap with default params"
    _use_neighbor_leaf_trick = True
    _p2m_class = None

class VectorizedLinearTreeSHAPSimpleApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap, Simple p2m, no neighbor-leaf trick."""

    name = "Woodelf Vec Simple"
    method = WOODELF_VEC_SIMPLE
    description = "vectorized_linear_tree_shap(p2m=Simple, neighbor_leaf_trick=False)"
    _use_neighbor_leaf_trick = False
    _p2m_class = LinearTreeShapPathToMatricesSimple


class VectorizedLinearTreeSHAPSimpleNLTApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap, Simple p2m, with neighbor-leaf trick."""

    name = "Woodelf Vec Simple + NLT"
    method = WOODELF_VEC_SIMPLE_NLT
    description = "vectorized_linear_tree_shap(p2m=Simple, neighbor_leaf_trick=True)"
    _use_neighbor_leaf_trick = True
    _p2m_class = LinearTreeShapPathToMatricesSimple


class VectorizedLinearTreeSHAPImprovedApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap, Improved p2m, no neighbor-leaf trick."""

    name = "Woodelf Vec Improved"
    method = WOODELF_VEC_IMPROVED
    description = "vectorized_linear_tree_shap(p2m=Improved, neighbor_leaf_trick=False)"
    _use_neighbor_leaf_trick = False
    _p2m_class = LinearTreeShapPathToMatricesImproved


class VectorizedLinearTreeSHAPImprovedNLTApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap, Improved p2m, with neighbor-leaf trick."""

    name = "Woodelf Vec Improved + NLT"
    method = WOODELF_VEC_IMPROVED_NLT
    description = "vectorized_linear_tree_shap(p2m=Improved, neighbor_leaf_trick=True)"
    _use_neighbor_leaf_trick = True
    _p2m_class = LinearTreeShapPathToMatricesImproved


class VectorizedLinearTreeSHAPDefaultApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap, default p2m, no neighbor-leaf trick."""

    name = "Woodelf Vec Default"
    method = WOODELF_VEC_DEFAULT
    description = "vectorized_linear_tree_shap(neighbor_leaf_trick=False)"
    _use_neighbor_leaf_trick = False
    _p2m_class = None


class VectorizedLinearTreeSHAPDefaultNLTApproach(VectorizedLinearTreeSHAPBase):
    """vectorized_linear_tree_shap, default p2m, with neighbor-leaf trick."""

    name = "Woodelf Vec Default + NLT"
    method = WOODELF_VEC_DEFAULT_NLT
    description = "vectorized_linear_tree_shap(neighbor_leaf_trick=True)"
    _use_neighbor_leaf_trick = True
    _p2m_class = None


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

def PathDependentSHAPTask(
    methods: list | None = None,
    extra_approaches: list[Approach] | None = None,
    n_repeats: int = 3,
    cache_root: Path = Path("cache"),
) -> Task:
    """
    Convenience factory that returns a Task preconfigured for path-dependent SHAP.

    Parameters
    ----------
    methods : list[Method] | None
        Which built-in methods to include.  Defaults to all built-in methods.
        Pass a subset to run only selected methods, e.g. ``methods=[SHAP]``.
    extra_approaches : list[Approach] | None
        Additional Approach instances (e.g. a custom method) appended after
        the built-in ones.
    """
    _registry = {
        SHAP:                       SHAPTreePathDependentApproach,
        WOODELF:                    WoodelfSHAPTreePathDependentApproach,
        VECTORIZED_LINEAR_TREE_SHAP: VectorizedLinearTreeSHAP,
        WOODELF_VEC_SIMPLE:         VectorizedLinearTreeSHAPSimpleApproach,
        WOODELF_VEC_SIMPLE_NLT: VectorizedLinearTreeSHAPSimpleNLTApproach,
        WOODELF_VEC_IMPROVED:   VectorizedLinearTreeSHAPImprovedApproach,
        WOODELF_VEC_IMPROVED_NLT: VectorizedLinearTreeSHAPImprovedNLTApproach,
        WOODELF_VEC_DEFAULT:    VectorizedLinearTreeSHAPDefaultApproach,
        WOODELF_VEC_DEFAULT_NLT: VectorizedLinearTreeSHAPDefaultNLTApproach,
    }
    if methods is None:
        methods = [SHAP, WOODELF]

    approaches: list[Approach] = [_registry[m]() for m in methods if m in _registry]
    if extra_approaches:
        approaches.extend(extra_approaches)

    return Task(
        name="Path-Dependent SHAP",
        approaches=approaches,
        n_repeats=n_repeats,
        cache_root=cache_root,
    )
