"""
Model-parsing benchmark as a treebranchmarks Experiment.

Compares six ways of turning a trained tree ensemble into an explainer-ready in-memory
structure — treelite, shap, shapiq, woodelf (via shap), woodelf (via treelite), and the
in-repo treelite -> shapiq converter — and renders the interactive HTML report.

Run from the project root:
    python -m benchmarks.model_parsing_experiment

Missions (each has exactly one free variable)
---------------------------------------------
``parsing: sweep model type``       ensemble_type varies; XGBoost, LightGBM, RandomForest,
                                    GradientBoosting, HistGradientBoosting, DecisionTree at
                                    a fixed 100 trees / depth 6. This is the *coverage*
                                    mission — it is where unsupported combinations show up.
``parsing: {model} — sweep T``      n_estimators over 10 / 50 / 100 / 500 / 1000.
``parsing: {model} — sweep D``      max_depth over 2 / 4 / 6 / 8 / 10 / 12 / 14 / 16.

The two scale sweeps are repeated per backend — XGBoost, LightGBM and RandomForest get a
mission each — so every mission varies exactly one thing. Folding all three into one sweep
would leave ensemble_type varying alongside T or D, which both muddies the report's x-axis
and averages three very different parsers' behaviour into one line.

Runtime
-------
~15 minutes. Parse time depends on tree structure, not row count, so the full California
Housing set is plenty. Two guards keep the tail bounded: Task stops repeating any parse that
exceeds 10 s, and ``ModelParsingApproach.BUDGET_S`` retires a parser from models larger than
one that already cost more than 20 s. Retired configurations are recorded as runtime errors
and score 0 rather than being dropped.

Depths stop at 16 rather than going unlimited. ``ModelConfig`` passes ``max_depth`` straight
to the estimator, and the three libraries spell "no limit" three different ways (XGBoost 0,
LightGBM -1, sklearn None), so an unlimited point cannot be expressed as one shared sweep
value. Depth 16 already saturates against the row count, which is the interesting regime.
"""

from pathlib import Path

from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.datasets import CaliforniaHousingDataset
from treebranchmarks.methods.model_parsing_approaches import ModelParsingTask
from treebranchmarks.models import (
    DecisionTreeWrapper,
    GradientBoostingWrapper,
    HistGradientBoostingWrapper,
    LightGBMWrapper,
    RandomForestWrapper,
    XGBoostWrapper,
)

CACHE_ROOT = Path("cache")
RESULTS_DIR = Path("results")

TASK_TYPE = "regression"          # California Housing
FIXED_TREES = 100
FIXED_DEPTH = 6

# Parsing time is a property of the fitted trees, so only one row is needed to drive the
# task; n and m are not free variables in any of these missions.
N_VALUES = [1]
M_VALUES = [0]

# One representative per backend: a booster with its own on-disk format, a second such
# booster, and sklearn's in-memory arrays. That is what actually distinguishes the parsers.
SWEEP_ENSEMBLES = (
    EnsembleType.XGBOOST,
    EnsembleType.LIGHTGBM,
    EnsembleType.RANDOM_FOREST,
)

TREE_COUNTS = [10, 50, 100, 500, 1000]
# Runs past the point where shapiq stops being usable on purpose. shapiq needs ~52 s for
# one depth-12 XGBoost model; at 14 and 16 it is refused by the approach's time budget and
# recorded as a runtime error, which scores 0 — the honest result for a parser too slow to
# use, and more informative than leaving the configuration out of the sweep.
DEPTHS = [2, 4, 6, 8, 10, 12, 14, 16]

_WRAPPERS = {
    EnsembleType.XGBOOST: XGBoostWrapper,
    EnsembleType.LIGHTGBM: LightGBMWrapper,
    EnsembleType.RANDOM_FOREST: RandomForestWrapper,
    EnsembleType.GRADIENT_BOOSTING: GradientBoostingWrapper,
    EnsembleType.HIST_GRADIENT_BOOSTING: HistGradientBoostingWrapper,
    EnsembleType.DECISION_TREE: DecisionTreeWrapper,
}

# HistGradientBoosting spells n_estimators "max_iter"; DecisionTree has neither.
_TREES_KEY = {
    EnsembleType.HIST_GRADIENT_BOOSTING: "max_iter",
    EnsembleType.DECISION_TREE: None,
}


def _model(ensemble: EnsembleType, n_trees: int = FIXED_TREES, depth: int = FIXED_DEPTH) -> dict:
    """One model config + its wrapper, keyed the way MissionConfig expects."""
    hyperparams: dict = {"max_depth": depth}
    trees_key = _TREES_KEY.get(ensemble, "n_estimators")
    if trees_key is not None:
        hyperparams[trees_key] = n_trees
    config = ModelConfig(ensemble_type=ensemble, hyperparams=hyperparams)
    return {config: _WRAPPERS[ensemble](task_type=TASK_TYPE)}


def build_experiment() -> Experiment:
    california = CaliforniaHousingDataset(cache_root=CACHE_ROOT)
    parsing_task = ModelParsingTask(n_repeats=5, cache_root=CACHE_ROOT)

    def mission(name: str, model_wrappers: dict) -> Mission:
        return Mission(MissionConfig(
            name=name,
            dataset=california,
            model_wrappers=model_wrappers,
            tasks=[parsing_task],
            n_values=N_VALUES,
            m_values=M_VALUES,
            cache_root=CACHE_ROOT,
        ))

    # ------------------------------------------------------------------
    # Coverage: every model family the repo can train, at fixed size.
    # ------------------------------------------------------------------
    sweep_model_type = mission(
        "parsing: sweep model type",
        {k: v for ensemble in _WRAPPERS for k, v in _model(ensemble).items()},
    )

    # ------------------------------------------------------------------
    # Scale sweeps — one mission per backend, so each has exactly one free
    # variable. Mixing ensembles into a single sweep would leave the report's
    # x-axis fighting with a second varying dimension.
    # ------------------------------------------------------------------
    scale_missions = []
    for ensemble in SWEEP_ENSEMBLES:
        label = ensemble.value

        # Ensemble size, at fixed depth.
        scale_missions.append(mission(
            f"parsing: {label} — sweep T (ensemble size)",
            {k: v for t in TREE_COUNTS for k, v in _model(ensemble, n_trees=t).items()},
        ))

        # Depth, at fixed ensemble size. Node count grows super-linearly.
        scale_missions.append(mission(
            f"parsing: {label} — sweep D (max depth)",
            {k: v for d in DEPTHS for k, v in _model(ensemble, depth=d).items()},
        ))

    return Experiment(
        name="model_parsing_v1",
        missions=[sweep_model_type, *scale_missions],
        results_dir=RESULTS_DIR,
        force_rerun=False,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
