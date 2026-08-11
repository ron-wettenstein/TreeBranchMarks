"""
WoodelfIQ — Woodelf across all six any-order tasks, interventional and path-dependent.

The Woodelf counterpart to
:class:`~treebranchmarks.methods.shapiq_method.ShapiqApproach`, and its mirror
image in shape: one class, two feature-perturbation modes, three interaction
orders each.  Used by ``benchmarks/woodelf_vs_shapiq_experiment.py`` and
``benchmarks/woodelf_vs_shapiq_path_dependent_experiment.py``.

============================================ ================== =============================================
Task                                         Mode               Woodelf metric
============================================ ================== =============================================
``background_shap``                          interventional     ``ShapleyValues()``
``background_shap_interactions``             interventional     ``ShapleyInteractionValues()``
``background_shap_interactions_order_3``     interventional     ``GeneralShapleyInteractionValues(1, 3)``
``path_dependent_shap``                      path-dependent     ``ShapleyValues()``
``path_dependent_interactions``              path-dependent     ``ShapleyInteractionValues()``
``path_dependent_interactions_order_3``      path-dependent     ``GeneralShapleyInteractionValues(1, 3)``
============================================ ================== =============================================

Mode is carried entirely by ``X_background``: present means interventional,
``None`` means path-dependent — the same switch ``WoodelfExplainer`` itself makes
under ``feature_perturbation="auto"``.

Why not reuse ``WoodelfApproach``?
----------------------------------
``WoodelfApproach`` goes through ``WoodelfExplainer.shap_values`` /
``.shap_interaction_values``, whose output formatter only knows how to build
pair-keyed columns (``woodelf/explainer.py::_output_formatting`` filters on
``(f1, f2)``).  Order-3 keys are 3-tuples, so that formatter silently drops every
column with ``exclude_zero_contribution_features=True`` and raises
``"cannot handle a non-unique multi-index"`` without it.  The *computation* is
fine — only the DataFrame assembly is order-2-only.

So every task here runs the same two calls ``calc_metric`` makes internally:

    load_decision_tree_ensemble_model(model, columns)   # parse
    hybrid_woodelf(ensemble, X_explain, X_background, metric)

which keeps all six tasks on one identical code path and sidesteps the formatter
entirely.  Both the parse and the computation are timed, mirroring
``ShapiqApproach``, which likewise counts ``TreeExplainer`` construction.

Agreement with shapiq
---------------------
Verified before either benchmark was written (XGBoost, California Housing,
``D=4``, ``T=20``):

===== ============================ ============================= ==============
order shapiq equivalent            interventional (n=2, m=50)    path-dependent (n=3)
===== ============================ ============================= ==============
1     ``max_order=1, index="SV"``  1.7e-08                       2.1e-07
2     ``max_order=2, index="SII"`` 4.8e-09                       2.5e-08
3     ``max_order=3, index="SII"`` 1.2e-09                       1.4e-08
===== ============================ ============================= ==============

Order 2 carries Woodelf's ``shap``-package convention (each pair halved and
mirrored), so its values are shapiq's SII divided by two — the same quantity,
scaled.  Order 3 uses ``min_order=1`` because shapiq's ``max_order=3`` computes
every order up to 3 internally regardless of ``min_order``; asking Woodelf for
orders 1–3 keeps the two doing equal work.

Routing note (path-dependent order 3)
-------------------------------------
``woodelf_sparse`` asserts that ``CardinalityInteractionIndicesMetric`` metrics
(which is what ``GeneralShapleyInteractionValues`` is) are background-only, so
order 3 cannot take the sparse path without a background set.  ``hybrid_woodelf``
handles that on its own: ``use_sparse_approach`` checks the metric against
``_SUPPORTED_SPARSE_PATH_DEPENDENT_METRICS``, which lists only ``ShapleyValues``,
``BanzhafValues`` and ``ShapleyInteractionValues``, so path-dependent order 3
routes to ``woodelf_for_high_depth`` instead and the assert is never reached.
Nothing to special-case here, but it is why path-dependent order 3 can behave
differently from orders 1–2 as depth grows.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import WOODELF


class WoodelfIQ(Approach):
    """Woodelf SHAP / interaction values for orders 1, 2 and 3, in either mode."""

    name = "Woodelf"
    method = WOODELF
    description = (
        "Woodelf any-order values via hybrid_woodelf (the code path "
        "WoodelfExplainer.calc_metric uses internally), interventional or "
        "path-dependent. Model parsing is included in the timing."
    )

    GPU = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metric(order: int):
        from woodelf.core.cube_metric import (
            GeneralShapleyInteractionValues,
            ShapleyInteractionValues,
            ShapleyValues,
        )

        if order == 1:
            return ShapleyValues()
        if order == 2:
            return ShapleyInteractionValues()
        return GeneralShapleyInteractionValues(min_order=1, max_order=order)

    def _run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        order: int,
    ) -> ApproachOutput:
        """Time parse + compute.  ``X_background=None`` selects path-dependent mode."""
        from woodelf.core.trees.parse_models import load_decision_tree_ensemble_model
        from woodelf.woodelf_sparse import hybrid_woodelf

        metric = self._metric(order)

        t0 = time.perf_counter()
        ensemble = load_decision_tree_ensemble_model(
            trained_model.raw_model, list(X_explain.columns)
        )
        hybrid_woodelf(
            ensemble, X_explain, X_background, metric,
            GPU=self.GPU, model_was_loaded=True,
        )
        return ApproachOutput(elapsed_s=time.perf_counter() - t0)

    def _run_interventional(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        order: int,
    ) -> ApproachOutput:
        """
        As ``_run``, but refuses a missing background set.

        Without this guard a background task invoked with ``m = 0`` would fall
        through to path-dependent mode and silently report the wrong quantity's
        runtime under an interventional task name.
        """
        if X_background is None:
            raise ValueError(
                "Interventional Woodelf tasks require X_background; got None "
                "(check the mission's m_values)."
            )
        return self._run(trained_model, X_explain, X_background, order)

    # ------------------------------------------------------------------
    # Task methods — interventional
    # ------------------------------------------------------------------

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run_interventional(trained_model, X_explain, X_background, order=1)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run_interventional(trained_model, X_explain, X_background, order=2)

    def background_shap_interactions_order_3(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run_interventional(trained_model, X_explain, X_background, order=3)

    # ------------------------------------------------------------------
    # Task methods — path-dependent
    # ------------------------------------------------------------------

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, None, order=1)

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, None, order=2)

    def path_dependent_interactions_order_3(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, None, order=3)


class WoodelfIQGPU(WoodelfIQ):
    """WoodelfIQ with GPU=True (requires CuPy: pip install cupy)."""

    name = "Woodelf GPU"
    description = (
        "Woodelf any-order values via hybrid_woodelf, accelerated on GPU (CuPy required)."
    )
    GPU = True
