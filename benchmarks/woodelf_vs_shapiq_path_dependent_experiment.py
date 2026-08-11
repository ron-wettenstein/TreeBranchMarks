"""
Woodelf vs shapiq — path-dependent SHAP and interaction values.

The path-dependent sibling of ``woodelf_vs_shapiq_experiment.py``.  Same
question — where is the turn-over point? — same datasets, same three orders,
same budget machinery.  There is no ``m``: path-dependent explainers read
coverage off the tree itself and take no background set, so ``n`` is the only
data-size axis.

What is compared
----------------
========================================== ============================================== ==========================================
Task                                       Woodelf                                        shapiq
========================================== ============================================== ==========================================
Path-dependent SHAP (order 1)              ``ShapleyValues()``                            ``max_order=1, index="SV"``
Path-dependent interactions (order 2)      ``ShapleyInteractionValues()``                 ``max_order=2, index="SII"``
Path-dependent interactions (order 3)      ``GeneralShapleyInteractionValues(1, 3)``      ``max_order=3, index="SII"``
========================================== ============================================== ==========================================

Verified to compute the same numbers before this benchmark was written
(XGBoost, California Housing, ``D=4``, ``T=20``, ``n=3``): order 1 to
``2.1e-07``, order 2 to ``2.5e-08`` after undoing Woodelf's ``shap``-package
halving convention, order 3 to ``1.4e-08``.

What the calibration found
--------------------------
The two libraries have opposite cost shapes here.  Woodelf is dominated by a
**fixed** cost (parse the model, build its tree structures) and is then nearly
flat in ``n``; shapiq builds cheaply — usually — and pays **per row**.  So the
crossing is simply where shapiq's per-row cost has eaten Woodelf's fixed cost,
which makes it a single threshold in ``n`` rather than the ``n × m`` curve the
interventional experiment traces.

Measured at ``D=6``, ``T=100`` (recorded in ``_CALIBRATION`` below):

============== ===== ========= ============ ============ ==========
Dataset        Order Woodelf   shapiq build shapiq /row  crossing n
============== ===== ========= ============ ============ ==========
California     1     1.35 s    0.08 s       0.0103 s     ~120
California     2     1.42 s    0.41 s       1.56 s       <1
California     3     1.66 s    0.75 s       2.33 s       <1
Breast Cancer  1     0.19 s    0.05 s       0.0100 s     ~14
Breast Cancer  2     0.19 s    0.14 s       0.12 s       <1
Breast Cancer  3     0.21 s    0.21 s       0.19 s       <1
Fraud          1     1.50 s    0.11 s       0.033 s      ~42
Fraud          2     1.80 s    6.0 s        2.0 s        <1
Fraud          3     2.20 s    105 s        33 s         <1
============== ===== ========= ============ ============ ==========

Only **order 1** has a crossing inside the measurable range.  At orders 2 and 3
Woodelf already leads at ``n = 1`` on every dataset — and on Fraud it leads
before a single row is explained, because shapiq's TreeSHAP-IQ construction over
397 features costs 6 s at order 2 and 105 s at order 3, both more than Woodelf's
entire run.  The sweeps are sized accordingly: order 1 gets a wide ``n`` sweep to
bracket its crossing, orders 2 and 3 get a short one to show the gap and its
direction, since marching to ``n = 10 000`` there would only buy a more expensive
restatement of the same verdict.

Sweeps
------
**n-sweep missions** — one per (dataset, order) at ``D = 6``.  ``n`` is the only
free variable, so each reads directly as "Woodelf takes the lead from n = X".

**Depth missions** — ``D ∈ {4, 6, 9, 12}`` at the n closest to that
(dataset, order)'s crossing.  Skipped where shapiq's **build cost alone** already
exceeds Woodelf's total run: the crossing there sits off the left edge of the n
axis, no depth can push it back on, and paying that setup four times over buys
nothing.  That rule drops Fraud orders 2 and 3 — the latter would have cost
~40 min to re-confirm a 50× Woodelf win.

Run from the project root::

    python -m benchmarks.woodelf_vs_shapiq_path_dependent_experiment

    # only re-time one side
    python -m benchmarks.woodelf_vs_shapiq_path_dependent_experiment --method shapiq
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
SWEEP_DEPTH = 6                     # depth used for the n-sweep missions
DEPTH_VALUES = [4, 6, 9, 12]        # depth sweep

#: Wall-clock budget for shapiq's per-row loop before it extrapolates in n.
SHAPIQ_TIME_BUDGET_S = 45.0
#: Rows explained past the budget before extrapolating (cheap rows only).
SHAPIQ_MIN_INSTANCES = 3

N_REPEATS = 3   # Task skips repeats once a run exceeds 10 s, so slow cells still run once

#: How far past the crossing an n value may sit and still be worth measuring.
_INCLUDE_FACTOR = 30
#: Never emit a mission with fewer points than this — at orders 2 and 3 the
#: crossing sits below n = 1, so the include rule alone would leave nothing.
_MIN_POINTS = 3

_ORDER_TASKS = [
    (1, TaskType.PATH_DEPENDENT_SHAP),
    (2, TaskType.PATH_DEPENDENT_INTERACTIONS),
    (3, TaskType.PATH_DEPENDENT_INTERACTIONS_ORDER_3),
]


@dataclass(frozen=True)
class _Calibration:
    """
    Measured cost shape for one (dataset, order) at ``D=6``, ``T=100``.

    Used only to *choose which cells to run* — every reported time is measured
    independently of these numbers.
    """

    crossing_n: float       # n at which the two methods cost the same
    shapiq_build_s: float   # shapiq's fixed explainer-construction cost
    woodelf_s: float        # Woodelf's near-flat total over this n range

    @property
    def depth_sweep_is_informative(self) -> bool:
        """
        Whether sweeping depth can still change the verdict.

        False once shapiq's fixed build cost alone *exceeds* Woodelf's whole run:
        the crossing is then off the left edge of the n axis, depth only pushes
        it further left, and each extra depth re-pays that build for nothing.

        The comparison is deliberately non-strict, so a tie still runs — where the
        two setups cost the same, depth is exactly what decides the winner.
        """
        return self.shapiq_build_s <= self.woodelf_s


_CALIBRATION = {
    ("california_housing", 1): _Calibration(crossing_n=120,  shapiq_build_s=0.08,  woodelf_s=1.35),
    ("california_housing", 2): _Calibration(crossing_n=0.6,  shapiq_build_s=0.41,  woodelf_s=1.42),
    ("california_housing", 3): _Calibration(crossing_n=0.4,  shapiq_build_s=0.75,  woodelf_s=1.66),
    ("breast_cancer", 1):      _Calibration(crossing_n=14,   shapiq_build_s=0.05,  woodelf_s=0.19),
    ("breast_cancer", 2):      _Calibration(crossing_n=0.4,  shapiq_build_s=0.14,  woodelf_s=0.19),
    ("breast_cancer", 3):      _Calibration(crossing_n=0.3,  shapiq_build_s=0.21,  woodelf_s=0.21),
    ("fraud_detection", 1):    _Calibration(crossing_n=42,   shapiq_build_s=0.11,  woodelf_s=1.50),
    ("fraud_detection", 2):    _Calibration(crossing_n=0.3,  shapiq_build_s=6.0,   woodelf_s=1.80),
    ("fraud_detection", 3):    _Calibration(crossing_n=0.02, shapiq_build_s=105.0, woodelf_s=2.20),
}

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
    """A dataset plus the n values that fit inside it."""

    def __init__(self, label: str, dataset, task_type: str, n_values: list[int]) -> None:
        self.label = label
        self.dataset = dataset
        self.task_type = task_type      # "classification" | "regression"
        self.n_values = n_values

    def wrapper(self) -> XGBoostWrapper:
        return XGBoostWrapper(task_type=self.task_type)

    def calibration(self, order: int) -> _Calibration:
        return _CALIBRATION[(self.label, order)]


def _dataset_specs() -> list[_DatasetSpec]:
    return [
        _DatasetSpec(
            "california_housing", CaliforniaHousingDataset(), "regression",
            n_values=[1, 10, 100, 1_000, 10_000],
        ),
        # 569 rows — n stops at 250 rather than recording an n it cannot sample.
        _DatasetSpec(
            "breast_cancer", BreastCancerDataset(), "classification",
            n_values=[1, 10, 100, 250],
        ),
        _DatasetSpec(
            "fraud_detection", FraudDetectionDataset(), "classification",
            n_values=[1, 10, 100, 1_000, 10_000],
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

def _n_values_near_crossing(spec: _DatasetSpec, order: int) -> list[int]:
    """
    The n values worth measuring: those up to ``_INCLUDE_FACTOR`` past the
    crossing, and never fewer than ``_MIN_POINTS``.

    Beyond that factor Woodelf wins by more than an order of magnitude — each
    further point costs shapiq a full time budget to restate a settled verdict.
    The floor matters at orders 2 and 3, where the crossing sits below n = 1 and
    the rule alone would select nothing at all.
    """
    cap = spec.calibration(order).crossing_n * _INCLUDE_FACTOR
    kept = [n for n in spec.n_values if n <= cap]
    if len(kept) < _MIN_POINTS:
        kept = spec.n_values[:_MIN_POINTS]
    return kept


def _crossing_n(spec: _DatasetSpec, order: int) -> int:
    """The available n value sitting closest to the crossing (log scale)."""
    target = max(spec.calibration(order).crossing_n, 1.0)
    return min(spec.n_values, key=lambda n: abs(math.log(n / target)))


def _n_sweep_missions(spec: _DatasetSpec) -> list[Mission]:
    """One mission per interaction order, sweeping n across the crossing."""
    model_wrappers = {_xgb_config(SWEEP_DEPTH): spec.wrapper()}

    missions = []
    for order, task_type in _ORDER_TASKS:
        missions.append(Mission(MissionConfig(
            dataset=spec.dataset,
            model_wrappers=model_wrappers,
            tasks=[_task(task_type)],
            n_values=_n_values_near_crossing(spec, order),
            m_values=[0],     # path-dependent: no background set
            name=f"{spec.label} order-{order} sweep_n (D={SWEEP_DEPTH})",
            cache_root=CACHE_ROOT,
        )))
    return missions


def _depth_missions(spec: _DatasetSpec) -> list[Mission]:
    """
    One mission per interaction order, sweeping D at the crossing n.

    Orders whose crossing is already decided by shapiq's build cost are skipped
    — see ``_Calibration.depth_sweep_is_informative``.
    """
    model_wrappers = {_xgb_config(d): spec.wrapper() for d in DEPTH_VALUES}

    missions = []
    for order, task_type in _ORDER_TASKS:
        if not spec.calibration(order).depth_sweep_is_informative:
            continue
        n = _crossing_n(spec, order)
        missions.append(Mission(MissionConfig(
            dataset=spec.dataset,
            model_wrappers=model_wrappers,
            tasks=[_task(task_type)],
            n_values=[n],
            m_values=[0],
            name=f"{spec.label} order-{order} sweep_D (n={n})",
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
    / "woodelf_vs_shapiq_path_dependent_summary.html"
)


def build_experiment() -> Experiment:
    return Experiment(
        name="woodelf_vs_shapiq_path_dependent_experiment",
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
