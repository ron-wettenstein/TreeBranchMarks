"""
ShapiqApproach — shapiq.TreeExplainer, interventional and path-dependent.

Covers six tasks, three per mode:

============================================ ================== =============================
Task                                         shapiq ``mode``    settings
============================================ ================== =============================
``background_shap``                          interventional     ``max_order=1, index="SV"``
``background_shap_interactions``             interventional     ``max_order=2, index="SII"``
``background_shap_interactions_order_3``     interventional     ``max_order=3, index="SII"``
``path_dependent_shap``                      pathdependent      ``max_order=1, index="SV"``
``path_dependent_interactions``              pathdependent      ``max_order=2, index="SII"``
``path_dependent_interactions_order_3``      pathdependent      ``max_order=3, index="SII"``
============================================ ================== =============================

Timing model
------------
Both modes explain **one instance at a time** — ``explain_X`` is a plain Python
loop over rows — so total cost is ``build_explainer + n × per_instance_cost``.
Per-instance cost is stable across rows, which is what makes the budget below
safe.  Interventional per-instance cost additionally scales with ``m``;
path-dependent has no ``m`` at all.

Time budget
-----------
Interventional shapiq costs roughly ``O(n × m)``, and path-dependent order 2/3
costs over a second per row even on an 8-feature model.  Either way the far end
of an n-sweep would run for hours.  Rather than skipping such cells outright
(which loses the data point the turn-over analysis needs) this approach explains
rows one at a time until ``time_budget_s`` is spent, then extrapolates linearly
in ``n``:

    total = build_time + (loop_time / rows_done) × n

and marks the result ``is_estimated=True``.  Extrapolation is linear in the one
dimension that really is linear; ``m`` is never extrapolated, since the full
background set is always passed to the explainer.

Worst-case cost of a single cell is therefore ``build_time + one instance``: one
row must always be explained to have anything to extrapolate from, and a row that
blows the whole budget on its own stops the loop immediately.  ``min_instances``
only holds the loop open while rows are *cheap*, where a single sample would be
noise.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from treebranchmarks.core.approach import Approach, ApproachOutput
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.methods.builtin import SHAPIQ


class ShapiqApproach(Approach):
    """shapiq.TreeExplainer for orders 1, 2 and 3, interventional or path-dependent."""

    name = "shapiq"
    method = SHAPIQ
    description = (
        "shapiq.TreeExplainer. Explains one row at a time; runs are capped by a "
        "wall-clock budget and extrapolated linearly in n."
    )

    def __init__(
        self,
        time_budget_s: float = 60.0,
        min_instances: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        time_budget_s : float
            Wall-clock budget for the per-row explanation loop.  Once exceeded,
            the remaining rows are extrapolated instead of explained.
        min_instances : int
            Keep explaining rows up to this count even after the budget is spent,
            so extrapolation never rests on a single noisy sample.  Ignored once a
            single row costs more than the whole budget — there, one sample is
            already decisive and a second would only double the bill.
        """
        self.time_budget_s = time_budget_s
        self.min_instances = max(1, min_instances)

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        max_order: int,
        mode: str = "interventional",
    ) -> ApproachOutput:
        import shapiq

        if mode == "interventional" and X_background is None:
            raise ValueError("Interventional shapiq tasks require X_background.")

        X_e = np.asarray(X_explain.to_numpy(dtype=np.float64))
        n = len(X_e)

        # Classification models are explained on the positive class, matching the
        # single-output raw margin Woodelf returns.
        class_index = 1 if hasattr(trained_model.raw_model, "predict_proba") else None

        # reference_dataset is meaningful only in interventional mode; path-dependent
        # reads the coverage off the tree itself and takes no background at all.
        reference_dataset = (
            np.asarray(X_background.to_numpy(dtype=np.float64))
            if mode == "interventional"
            else None
        )

        t0 = time.perf_counter()
        explainer = shapiq.TreeExplainer(
            model=trained_model.raw_model,
            mode=mode,
            reference_dataset=reference_dataset,
            max_order=max_order,
            index="SV" if max_order == 1 else "SII",
            class_index=class_index,
        )
        build_s = time.perf_counter() - t0

        loop_s = 0.0
        done = 0
        for i in range(n):
            t1 = time.perf_counter()
            explainer.explain(X_e[i])
            per_instance = time.perf_counter() - t1
            loop_s += per_instance
            done += 1

            # Stop once the budget is spent.  min_instances holds the loop open past
            # that point only while rows are cheap enough that a handful of them still
            # fits the budget — when the average row already costs more than its share
            # (budget / min_instances), further samples just multiply the bill without
            # sharpening an estimate that is already stable.
            if loop_s > self.time_budget_s and (
                done >= self.min_instances
                or loop_s / done > self.time_budget_s / self.min_instances
            ):
                break

        if done == n:
            return ApproachOutput(elapsed_s=build_s + loop_s)

        total = build_s + (loop_s / done) * n
        return ApproachOutput(
            elapsed_s=total,
            is_estimated=True,
            estimation_description=(
                f"explained {done} of {n} rows in {loop_s:.1f}s "
                f"(budget {self.time_budget_s:.0f}s), extrapolated linearly in n; "
                f"explainer build ({build_s:.1f}s) counted once"
            ),
        )

    # ------------------------------------------------------------------
    # Task methods
    # ------------------------------------------------------------------

    def background_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background, max_order=1)

    def background_shap_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background, max_order=2)

    def background_shap_interactions_order_3(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, X_background, max_order=3)

    def path_dependent_shap(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, None, max_order=1, mode="pathdependent")

    def path_dependent_interactions(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, None, max_order=2, mode="pathdependent")

    def path_dependent_interactions_order_3(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
    ) -> ApproachOutput:
        return self._run(trained_model, X_explain, None, max_order=3, mode="pathdependent")
