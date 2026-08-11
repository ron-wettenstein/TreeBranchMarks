"""
Woodelf vs shapiq — interventional SHAP and interaction values.

Goal
----
Find the **turn-over point**: the data size at which Woodelf becomes the faster
choice over ``shapiq.TreeExplainer``.  Everything here is interventional
(background) — no path-dependent tasks.

What is compared
----------------
==================================== ============================================ ==========================================
Task                                 Woodelf                                      shapiq
==================================== ============================================ ==========================================
Background SHAP (order 1)            ``ShapleyValues()``                          ``max_order=1, index="SV"``
Background interactions (order 2)    ``ShapleyInteractionValues()``               ``max_order=2, index="SII"``
Background interactions (order 3)    ``GeneralShapleyInteractionValues(1, 3)``    ``max_order=3, index="SII"``
==================================== ============================================ ==========================================

The two libraries were verified to compute the same numbers before this
benchmark was written (XGBoost, California Housing, ``D=4``, ``T=20``,
``n=2``, ``m=50``):

- order 1 — ``max |diff| = 1.7e-08``
- order 2 — ``max |diff| = 4.8e-09`` after undoing Woodelf's ``shap``-package
  halving convention (Woodelf reports each pair twice at half value)
- order 3 — ``max |diff| = 1.2e-09``

Sweeps
------
The experiment is trimmed to the **crossing band** — cells where Woodelf wins by
more than ~30× are not measured.  They cost shapiq a full time budget each and
say nothing about where the threshold is.  ``_CROSSING_NM`` records the measured
crossing per (dataset, order) and ``_INCLUDE_FACTOR`` how far past it to look.

**n-sweep missions** — one per (dataset, order, m) at ``D = 6``, sweeping n
across the crossing.  n is the single free variable, so each mission reads
directly as "at this m, Woodelf takes the lead from n = X onward".

**Depth missions** — ``D ∈ {4, 6, 9, 12}`` at ``m = 10`` and the n closest to
that (dataset, order)'s crossing, so the sweep shows how depth *moves* the
crossing rather than re-confirming a winner data size already decided.

Datasets
--------
============== ========= ========== =========================== ================
Dataset        Rows      Features   n values                    m values
============== ========= ========== =========================== ================
California     20 640    8          1, 10, 100, 1 000, 10 000   1, 10, 100, 1 000
Breast Cancer  569       30         1, 10, 100, 250             1, 10, 100, 250
Fraud          590 540   397        1, 10, 100, 1 000, 10 000   1, 10, 100, 1 000
============== ========= ========== =========================== ================

Not every (n, m) pair from those lists runs — only those near the crossing.

``m`` stops at 1 000: by ``m = 1 000`` shapiq already loses at ``n = 1`` on every
dataset, so an ``m = 10 000`` column could only re-confirm a Woodelf win, at
~107 s per cell — the most expensive cells in the whole experiment.

Breast Cancer holds 569 rows, so the requested 1 000 / 10 000 points cannot be
sampled without replacement.  Its values stop at 250 (``n + m = 500 ≤ 569``);
asking for more would silently record ``n = 10 000`` for a 569-row run.  It
contributes the "few rows, many features" corner instead.

Cost control
------------
Interventional shapiq is ``O(n × m)`` and explains one row at a time, so cells
past the crossing would run for hours.  :class:`ShapiqApproach` therefore
explains rows until ``SHAPIQ_TIME_BUDGET_S`` is spent and then extrapolates
linearly in ``n`` (marked ``is_estimated`` in the report, and still scored — an
extrapolated time is real performance data).  Woodelf is left uncapped; it never
needed a cap, the worst cell measured anywhere was 8.4 s (fraud,
``n = m = 10 000``, order 3).

Run from the project root::

    python -m benchmarks.woodelf_vs_shapiq_experiment

    # only re-time one side
    python -m benchmarks.woodelf_vs_shapiq_experiment --method shapiq
"""

from __future__ import annotations

import math
from pathlib import Path

from treebranchmarks import Experiment
from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks.core.mission import Mission, MissionConfig
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.datasets import CaliforniaHousingDataset, FraudDetectionDataset
from treebranchmarks.datasets.sklearn_datasets import BreastCancerDataset
from treebranchmarks.methods.shapiq_method import ShapiqApproach
from treebranchmarks.methods.woodelfiq_method import WoodelfIQ
from treebranchmarks.models.xgboost_model import XGBoostWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

N_TREES = 100
GRID_DEPTH = 6                      # depth used for the n-sweep missions
DEPTH_VALUES = [4, 6, 9, 12]        # depth sweep
DEPTH_SWEEP_M = 10                  # background size held fixed while D varies

#: Measured location of the crossing, expressed as the product ``n × m`` at which
#: shapiq and Woodelf cost the same (XGBoost, ``D=6``, ``T=100``).
#:
#: shapiq costs ``build(m) + n × per_row(m)`` with ``per_row`` very close to linear
#: in ``m``, while Woodelf is near-flat over this whole range.  Setting the two
#: equal makes the crossing a curve of constant ``n × m``, so one number per
#: (dataset, order) pins it down.  Derived from the calibration probes:
#:
#:   - California  ``per_row ≈ 0.0095 × m`` s, Woodelf ≈ 2.5 s  → n × m ≈ 300
#:   - Breast Cancer ``per_row ≈ 0.0055 × m`` s, Woodelf ≈ 0.3 s → n × m ≈ 60
#:   - Fraud order 1/2 ``per_row ≈ 0.006 × m`` s, Woodelf ≈ 2.5 s → n × m ≈ 400
#:   - Fraud order 3 ``per_row ≈ 0.00087 × m`` s, Woodelf ≈ 3.9 s → n × m ≈ 4500
#:
#: Fraud order 3 sits an order of magnitude higher because shapiq routes 397
#: features past its dense-buffer limit onto the sparse kernel, which is much
#: faster than its own order-2 path.
_CROSSING_NM = {
    ("california_housing", 1): 300,
    ("california_housing", 2): 300,
    ("california_housing", 3): 300,
    ("breast_cancer", 1): 60,
    ("breast_cancer", 2): 60,
    ("breast_cancer", 3): 60,
    ("fraud_detection", 1): 400,
    ("fraud_detection", 2): 400,
    ("fraud_detection", 3): 4500,
}

#: How far past the crossing a cell may sit and still be worth measuring.
#: Beyond ~30× the crossing product Woodelf wins by more than an order of
#: magnitude; confirming that costs shapiq a full time budget per cell and tells
#: us nothing about where the threshold is, so those cells are dropped.
_INCLUDE_FACTOR = 30

#: Wall-clock budget for shapiq's per-row loop before it extrapolates in n.
SHAPIQ_TIME_BUDGET_S = 45.0
#: Rows explained past the budget before extrapolating (cheap rows only — a row
#: that costs more than the whole budget stops the loop on its own).
SHAPIQ_MIN_INSTANCES = 3

N_REPEATS = 3   # Task skips repeats once a run exceeds 10 s, so slow cells still run once

_ORDER_TASKS = [
    (1, TaskType.BACKGROUND_SHAP),
    (2, TaskType.BACKGROUND_SHAP_INTERACTIONS),
    (3, TaskType.BACKGROUND_SHAP_INTERACTIONS_ORDER_3),
]

# ---------------------------------------------------------------------------
# Shared approach instances (stateless apart from the budget settings)
# ---------------------------------------------------------------------------

_WOODELF = WoodelfIQ()
_SHAPIQ = ShapiqApproach(
    time_budget_s=SHAPIQ_TIME_BUDGET_S,
    min_instances=SHAPIQ_MIN_INSTANCES,
)


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------

class _DatasetSpec:
    """A dataset plus the sweep values that fit inside it."""

    def __init__(
        self,
        label: str,
        dataset,
        task_type: str,
        n_values: list[int],
        m_values: list[int],
    ) -> None:
        self.label = label
        self.dataset = dataset
        self.task_type = task_type      # "classification" | "regression"
        self.n_values = n_values
        self.m_values = m_values

    def wrapper(self) -> XGBoostWrapper:
        return XGBoostWrapper(task_type=self.task_type)

    def crossing_nm(self, order: int) -> int:
        return _CROSSING_NM[(self.label, order)]


def _dataset_specs() -> list[_DatasetSpec]:
    # m stops at 1 000: by m = 1 000 shapiq already loses at n = 1 on every
    # dataset, so an m = 10 000 column could only re-confirm a Woodelf win — at
    # ~107 s per cell, the most expensive cells in the whole experiment.
    return [
        _DatasetSpec(
            "california_housing", CaliforniaHousingDataset(), "regression",
            n_values=[1, 10, 100, 1_000, 10_000],
            m_values=[1, 10, 100, 1_000],
        ),
        # 569 rows: n + m must stay within the dataset, so its values stop at 250.
        _DatasetSpec(
            "breast_cancer", BreastCancerDataset(), "classification",
            n_values=[1, 10, 100, 250],
            m_values=[1, 10, 100, 250],
        ),
        _DatasetSpec(
            "fraud_detection", FraudDetectionDataset(), "classification",
            n_values=[1, 10, 100, 1_000, 10_000],
            m_values=[1, 10, 100, 1_000],
        ),
    ]


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

def _xgb_config(depth: int) -> ModelConfig:
    return ModelConfig(
        ensemble_type=EnsembleType.XGBOOST,
        hyperparams={
            "max_depth": depth,
            "n_estimators": N_TREES,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
        },
        random_state=42,
    )


def _task(task_type: TaskType) -> Task:
    return Task(
        task_type=task_type,
        approaches=[_WOODELF, _SHAPIQ],
        n_repeats=N_REPEATS,
        cache_root=CACHE_ROOT,
    )


# ---------------------------------------------------------------------------
# Mission builders
# ---------------------------------------------------------------------------

def _n_values_near_crossing(spec: _DatasetSpec, order: int, m: int) -> list[int]:
    """
    The n values worth measuring at this m: those at or below the crossing, plus
    those up to ``_INCLUDE_FACTOR`` past it.

    Cells further out are dropped — Woodelf wins them by more than an order of
    magnitude, and each costs shapiq a full time budget to say so.
    """
    cap = spec.crossing_nm(order) * _INCLUDE_FACTOR
    return [n for n in spec.n_values if n * m <= cap]


def _crossing_n(spec: _DatasetSpec, order: int, m: int) -> int:
    """The available n value sitting closest to the crossing at this m (log scale)."""
    target = spec.crossing_nm(order) / m
    return min(spec.n_values, key=lambda n: abs(math.log(n / target)))


def _n_sweep_missions(spec: _DatasetSpec) -> list[Mission]:
    """
    One mission per (interaction order, m), sweeping n across the crossing.

    n is the single free variable, which is both what ``Mission`` is designed
    around and how the answer reads: "at this m, Woodelf takes the lead from
    n = X onward".
    """
    wrapper = spec.wrapper()
    model_wrappers = {_xgb_config(GRID_DEPTH): wrapper}

    missions = []
    for order, task_type in _ORDER_TASKS:
        for m in spec.m_values:
            n_values = _n_values_near_crossing(spec, order, m)
            if len(n_values) < 2:
                # A single point cannot show a crossing.
                continue
            missions.append(Mission(MissionConfig(
                dataset=spec.dataset,
                model_wrappers=model_wrappers,
                tasks=[_task(task_type)],
                n_values=n_values,
                m_values=[m],
                name=f"{spec.label} order-{order} sweep_n (m={m}, D={GRID_DEPTH})",
                cache_root=CACHE_ROOT,
            )))
    return missions


def _depth_missions(spec: _DatasetSpec) -> list[Mission]:
    """
    One mission per interaction order, sweeping D at a fixed (n, m).

    The point is picked at each (dataset, order)'s own crossing rather than at a
    shared (n, m), so the sweep shows how depth *moves* the crossing instead of
    re-confirming a winner already decided by data size.
    """
    wrapper = spec.wrapper()
    model_wrappers = {_xgb_config(d): wrapper for d in DEPTH_VALUES}

    missions = []
    for order, task_type in _ORDER_TASKS:
        n = _crossing_n(spec, order, DEPTH_SWEEP_M)
        missions.append(Mission(MissionConfig(
            dataset=spec.dataset,
            model_wrappers=model_wrappers,
            tasks=[_task(task_type)],
            n_values=[n],
            m_values=[DEPTH_SWEEP_M],
            name=(
                f"{spec.label} order-{order} sweep_D "
                f"(n={n}, m={DEPTH_SWEEP_M})"
            ),
            cache_root=CACHE_ROOT,
        )))
    return missions


def build_missions() -> list[Mission]:
    missions: list[Mission] = []
    specs = _dataset_specs()
    for spec in specs:
        missions += _n_sweep_missions(spec)
    for spec in specs:
        missions += _depth_missions(spec)
    return missions


_SUMMARY_HTML = (
    Path(__file__).parent.parent
    / "treebranchmarks" / "report" / "benchmark_summaries"
    / "woodelf_vs_shapiq_summary.html"
)


def build_experiment() -> Experiment:
    return Experiment(
        name="woodelf_vs_shapiq_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
        summary_html_path=_SUMMARY_HTML,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
