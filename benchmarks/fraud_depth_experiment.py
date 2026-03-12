"""
Fraud Detection depth-sweep experiment: compare Woodelf vs SHAP across tree depths.

Dataset / models
----------------
* IEEE-CIS Fraud Detection dataset (binary classification)
* LightGBM — two model groups:
    - Medium depth (T=10):   D = 9, 12, 15, 18, 21
    - High depth   (T=1):    D = 25, 30, 35, 40, 50, 60

Missions (10 total — one per task/depth-group combination)
----------------------------------------------------------
Each task type has one mission for medium depths and one for high depths:
  1+2.  PD SHAP    n=100 000
  3+4.  PD SHAP    n=1
  5+6.  BG SHAP    m=100,     n=400 000
  7+8.  BG SHAP    m=400 000, n=100 000
  9+10. BG SHAP IV m=400 000, n=10 000

The x-axis of each chart is D (depth), auto-detected from the per-model variation.

Run from the project root:
    python -m benchmarks.fraud_depth_experiment
"""

from pathlib import Path

from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import FraudDetectionDataset
from treebranchmarks.models import LightGBMWrapper
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.woodelf_method import WoodelfApproach

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

# ---------------------------------------------------------------------------
# Base hyperparameter templates
# ---------------------------------------------------------------------------

# Medium-depth model: 10 trees, enough leaves for depth up to ~21
_LGBM_MEDIUM_BASE = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 2024,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
    "force_col_wise": True,
    "n_estimators": 10,
    # max_depth filled per model below
}

# High-depth model: 1 tree, large leaf budget to allow very deep growth
_LGBM_HIGH_BASE = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 10000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
    "force_col_wise": True,
    "n_estimators": 1,
    # max_depth filled per model below
}

MEDIUM_DEPTHS = [9, 12, 15, 18]
HIGH_DEPTHS   = [20, 25, 30, 35, 40, 50, 60]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lgbm_models(base_params: dict, depths: list[int]) -> dict:
    """Build a {ModelConfig: LightGBMWrapper} dict, one entry per depth."""
    return {
        ModelConfig(
            ensemble_type=EnsembleType.LIGHTGBM,
            hyperparams={**base_params, "max_depth": d},
            random_state=42,
        ): LightGBMWrapper(task_type="regression")
        for d in depths
    }


# ---------------------------------------------------------------------------
# Mission builder
# ---------------------------------------------------------------------------

def build_missions(cache_root: Path = CACHE_ROOT, approaches=None) -> list[Mission]:
    if approaches is None:
        approaches = [SHAPApproach(), WoodelfApproach()]

    fraud = FraudDetectionDataset(cache_root=cache_root)

    medium_models = _lgbm_models(_LGBM_MEDIUM_BASE, MEDIUM_DEPTHS)
    high_models   = _lgbm_models(_LGBM_HIGH_BASE,   HIGH_DEPTHS)

    pd_shap_task = Task(TaskType.PATH_DEPENDENT_SHAP,          approaches, n_repeats=1, cache_root=cache_root)
    bg_shap_task = Task(TaskType.BACKGROUND_SHAP,              approaches, n_repeats=1, cache_root=cache_root)
    bg_iv_task   = Task(TaskType.BACKGROUND_SHAP_INTERACTIONS, approaches, n_repeats=1, cache_root=cache_root)

    missions = []
    for label, models in [("medium depth", medium_models), ("high depth", high_models)]:

        # 1/6 — PD SHAP n=100 000
        missions.append(Mission(MissionConfig(
            name=f"fraud: PD SHAP n=100k ({label})",
            dataset=fraud,
            model_wrappers=models,
            tasks=[pd_shap_task],
            n_values=[100_000],
            m_values=[0],
            cache_root=cache_root,
        )))

        # 2/7 — PD SHAP n=1
        missions.append(Mission(MissionConfig(
            name=f"fraud: PD SHAP n=1 ({label})",
            dataset=fraud,
            model_wrappers=models,
            tasks=[pd_shap_task],
            n_values=[1],
            m_values=[0],
            cache_root=cache_root,
        )))

        # 3/8 — BG SHAP m=100, n=400 000
        missions.append(Mission(MissionConfig(
            name=f"fraud: BG SHAP m=100 n=400k ({label})",
            dataset=fraud,
            model_wrappers=models,
            tasks=[bg_shap_task],
            n_values=[400_000],
            m_values=[100],
            cache_root=cache_root,
        )))

        # 4/9 — BG SHAP m=400 000, n=100 000
        missions.append(Mission(MissionConfig(
            name=f"fraud: BG SHAP m=400k n=100k ({label})",
            dataset=fraud,
            model_wrappers=models,
            tasks=[bg_shap_task],
            n_values=[100_000],
            m_values=[400_000],
            cache_root=cache_root,
        )))

        # 5/10 — BG SHAP IV m=400 000, n=10 000
        missions.append(Mission(MissionConfig(
            name=f"fraud: BG SHAP IV m=400k n=10k ({label})",
            dataset=fraud,
            model_wrappers=models,
            tasks=[bg_iv_task],
            n_values=[10_000],
            m_values=[400_000],
            cache_root=cache_root,
        )))

    return missions


def build_experiment() -> Experiment:
    return Experiment(
        name="fraud_depth_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
    )


if __name__ == "__main__":
    experiment = build_experiment()
    experiment.run()
    report_path = experiment.generate_html()
    print(f"\nOpen the report in your browser:\n  {report_path.resolve()}")
