"""
Vectorized Linear TreeSHAP comparison — Fraud Detection depth sweep.

Compares 7 path-dependent SHAP implementations across tree depths:
  - shap package (reference)
  - 6 vectorized_linear_tree_shap variants:
      Simple / Improved / Default  ×  neighbor-leaf trick on/off

Dataset / models
----------------
  * IEEE-CIS Fraud Detection dataset (binary classification)
  * LightGBM:
      - Medium depth (T=10): D = 9, 12, 15, 18
      - High depth   (T=1):  D = 20, 25, 30, 35, 40, 50, 60
  * XGBoost (no-pruning, T=1): D = 9, 12, 18, 21, 24

Missions
--------
LightGBM (medium + high depth), for n in [1, 10k, 100k]:
  - PD SHAP  m=0

XGBoost (no-pruning depth sweep):
  - PD SHAP  n=10 000, m=0

Run from the project root:
    python -m benchmarks.fraud_pd_shap_experiment
"""

from pathlib import Path

from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import FraudDetectionDataset
from treebranchmarks.models import LightGBMWrapper, XGBoostWrapper
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.linear_tree_shap_method import (
    VectorizedLinearTreeSHAPSimpleApproach,
    VectorizedLinearTreeSHAPSimpleNLTApproach,
    VectorizedLinearTreeSHAPImprovedApproach,
    VectorizedLinearTreeSHAPImprovedNLTApproach,
    VectorizedLinearTreeSHAPDefaultApproach,
    VectorizedLinearTreeSHAPDefaultNLTApproach,
    VectorizedLinearTreeSHAPRecursiveNLTApproach,
    VectorizedLinearTreeSHAPV6Approach,
)
from treebranchmarks.methods.linear_treeshap_v6_method import LinearTreeSHAPV6Approach

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

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
}

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
}

_XGB_BASE = {
    "objective": "reg:squarederror",
    "seed": 123,
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "subsample": 1,
    "colsample_bytree": 1,
}

MEDIUM_DEPTHS  = [9, 12, 15, 18]
HIGH_DEPTHS    = [20, 25, 30, 35, 40, 50, 60]
XGB_DEPTHS     = [9, 12, 18, 21, 24]

_VEC_APPROACHES = [
    SHAPApproach(),
    VectorizedLinearTreeSHAPRecursiveNLTApproach(),
    VectorizedLinearTreeSHAPV6Approach(),
    LinearTreeSHAPV6Approach(),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lgbm_models(base_params: dict, depths: list[int]) -> dict:
    return {
        ModelConfig(
            ensemble_type=EnsembleType.LIGHTGBM,
            hyperparams={**base_params, "max_depth": d},
            random_state=42,
        ): LightGBMWrapper(task_type="regression")
        for d in depths
    }


def _xgb_models(depths: list[int], n_estimators: int) -> dict:
    return {
        ModelConfig(
            ensemble_type=EnsembleType.XGBOOST,
            hyperparams={**_XGB_BASE, "max_depth": d, "n_estimators": n_estimators},
            random_state=42,
        ): XGBoostWrapper(task_type="regression")
        for d in depths
    }


# ---------------------------------------------------------------------------
# Mission builder
# ---------------------------------------------------------------------------

def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    fraud = FraudDetectionDataset(cache_root=cache_root)

    medium_models          = _lgbm_models(_LGBM_MEDIUM_BASE, MEDIUM_DEPTHS)
    high_models            = _lgbm_models(_LGBM_HIGH_BASE,   HIGH_DEPTHS)
    xgb_models             = _xgb_models(XGB_DEPTHS, n_estimators=10)
    xgb_single_tree_models = _xgb_models(XGB_DEPTHS, n_estimators=1)

    missions = []

    pd_task = Task(TaskType.PATH_DEPENDENT_SHAP, _VEC_APPROACHES, n_repeats=1, cache_root=cache_root)

    # LightGBM missions: n in [1, 100, 1000, 10_000, 100_000]
    for label, models in [("medium depth", medium_models), ("high depth", high_models)]:
        for n in [1, 100, 1000, 10_000, 20_000]:
            n_label = {1: "n=1", 100: "n=100", 1000: "n=1k", 10_000: "n=10k", 20_000: "n=20k"}[n]
            missions.append(Mission(MissionConfig(
                name=f"fraud vec PD SHAP {n_label} ({label})",
                dataset=fraud,
                model_wrappers=models,
                tasks=[pd_task],
                n_values=[n],
                m_values=[0],
                cache_root=cache_root,
            )))

    missions.append(Mission(MissionConfig(
        name="fraud vec PD SHAP n=100k (medium depth)",
        dataset=fraud,
        model_wrappers=medium_models,
        tasks=[pd_task],
        n_values=[100_000],
        m_values=[0],
        cache_root=cache_root,
    )))

    # XGBoost missions
    xgb_task = Task(TaskType.PATH_DEPENDENT_SHAP, _VEC_APPROACHES, n_repeats=1, cache_root=cache_root)
    for n in [1, 1_000, 10_000, 20_000]:
        n_label = {1: "n=1", 1_000: "n=1k", 10_000: "n=10k", 20_000: "n=20k"}[n]
        missions.append(Mission(MissionConfig(
            name=f"fraud vec PD SHAP {n_label} (XGBoost T=10)",
            dataset=fraud,
            model_wrappers=xgb_models,
            tasks=[xgb_task],
            n_values=[n],
            m_values=[0],
            cache_root=cache_root,
        )))

    # XGBoost mission: n=100k, T=1 depth sweep
    missions.append(Mission(MissionConfig(
        name="fraud vec PD SHAP n=100k (XGBoost T=1)",
        dataset=fraud,
        model_wrappers=xgb_single_tree_models,
        tasks=[xgb_task],
        n_values=[100_000],
        m_values=[0],
        cache_root=cache_root,
    )))

    return missions


def build_experiment() -> Experiment:
    return Experiment(
        name="fraud_pd_shap_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
