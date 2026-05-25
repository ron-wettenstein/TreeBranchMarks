"""
Hybrid Woodelf comparison — Fraud Detection depth sweep.

Compares 4 implementations across tree depths:
  - HybridWoodelfSparseApproach         (use_sparse_approaches=True)
  - HybridWoodelfSparseNoFastMNApproach (use_sparse_approaches=True, use_faster_mn_p2s=False)
  - HybridWoodelfAutoApproach           (use_sparse_approaches=False, auto per-leaf)
  - WoodelfHDApproach                   (baseline: woodelf_for_high_depth directly)

Dataset / models
----------------
  * IEEE-CIS Fraud Detection dataset (binary classification)
  * LightGBM:
      - Medium depth (T=10): D = 9, 12, 15, 18
      - High depth   (T=1):  D = 20, 25, 30, 35, 40, 50, 60
  * XGBoost (no-pruning, T=1): D = 9, 12, 18, 21, 24

Missions
--------
Path-dependent SHAP (m=0):
  - LightGBM medium/high depth, n in [1, 10k, 100k]
  - XGBoost T=10, n=10k

Background SHAP (m=100k):
  - LightGBM medium/high depth, n=10k
  - XGBoost T=10, n=10k
  (SparseNoFastMN vs Sparse difference is only visible here)

Path-dependent SHAP Interactions (m=0):
  - LightGBM medium/high depth, n in [1, 10k]
  - XGBoost T=10, n=10k

Run from the project root:
    python -m benchmarks.wip.fraud_hybrid_woodelf_experiment
"""

from pathlib import Path

from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import FraudDetectionDataset
from treebranchmarks.models import LightGBMWrapper, XGBoostWrapper
from treebranchmarks.core.approach import ApproachOutput
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.methods.woodelf_original_and_hd_method import WoodelfHDApproach
from treebranchmarks.methods.woodelf_hybrid_method import (
    HybridWoodelfSparseApproach,
    HybridWoodelfSparseNoFastMNApproach,
    HybridWoodelfAutoApproach,
)

_MEMORY_CRASH = ApproachOutput(elapsed_s=0.0, memory_crash=True)
_HD_CRASH_DEPTH = 18


class _WoodelfHDApproach(WoodelfHDApproach):
    """WoodelfHD with memory_crash=True for D > 18 (avoids OOM on high-depth trees)."""

    def _check(self, trained_model):
        return _MEMORY_CRASH if trained_model.params.D > _HD_CRASH_DEPTH else None

    def path_dependent_shap(self, trained_model, X_explain, X_background):
        return self._check(trained_model) or super().path_dependent_shap(trained_model, X_explain, X_background)

    def path_dependent_interactions(self, trained_model, X_explain, X_background):
        return self._check(trained_model) or super().path_dependent_interactions(trained_model, X_explain, X_background)

    def background_shap(self, trained_model, X_explain, X_background):
        return self._check(trained_model) or super().background_shap(trained_model, X_explain, X_background)

    def background_shap_interactions(self, trained_model, X_explain, X_background):
        return self._check(trained_model) or super().background_shap_interactions(trained_model, X_explain, X_background)

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

MEDIUM_DEPTHS = [9, 12, 15, 18]
HIGH_DEPTHS   = [20, 25, 30, 35, 40, 50, 60]
XGB_DEPTHS    = [9, 12, 18, 21, 24]

_APPROACHES = [
    HybridWoodelfSparseApproach(),
    HybridWoodelfSparseNoFastMNApproach(),
    HybridWoodelfAutoApproach(),
    _WoodelfHDApproach(),
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

    medium_models = _lgbm_models(_LGBM_MEDIUM_BASE, MEDIUM_DEPTHS)
    high_models   = _lgbm_models(_LGBM_HIGH_BASE,   HIGH_DEPTHS)
    xgb_models    = _xgb_models(XGB_DEPTHS, n_estimators=10)

    pd_task  = Task(TaskType.PATH_DEPENDENT_SHAP,         _APPROACHES, n_repeats=1, cache_root=cache_root)
    bg_task  = Task(TaskType.BACKGROUND_SHAP,             _APPROACHES, n_repeats=1, cache_root=cache_root)
    int_task = Task(TaskType.PATH_DEPENDENT_INTERACTIONS, _APPROACHES, n_repeats=1, cache_root=cache_root)

    missions = []

    # ------------------------------------------------------------------
    # Path-dependent SHAP (m=0)
    # ------------------------------------------------------------------
    for label, models in [("medium depth", medium_models), ("high depth", high_models)]:
        for n in [1, 10_000, 100_000]:
            n_label = {1: "n=1", 10_000: "n=10k", 100_000: "n=100k"}[n]
            missions.append(Mission(MissionConfig(
                name=f"fraud hybrid PD SHAP {n_label} ({label})",
                dataset=fraud,
                model_wrappers=models,
                tasks=[pd_task],
                n_values=[n],
                m_values=[0],
                cache_root=cache_root,
            )))

    missions.append(Mission(MissionConfig(
        name="fraud hybrid PD SHAP n=10k (XGBoost T=10)",
        dataset=fraud,
        model_wrappers=xgb_models,
        tasks=[pd_task],
        n_values=[10_000],
        m_values=[0],
        cache_root=cache_root,
    )))

    # ------------------------------------------------------------------
    # Background SHAP — SparseNoFastMN vs Sparse difference shows here
    # ------------------------------------------------------------------
    for label, models in [("medium depth", medium_models), ("high depth", high_models)]:
        missions.append(Mission(MissionConfig(
            name=f"fraud hybrid BG SHAP n=10k m=100k ({label})",
            dataset=fraud,
            model_wrappers=models,
            tasks=[bg_task],
            n_values=[10_000],
            m_values=[100_000],
            cache_root=cache_root,
        )))

    missions.append(Mission(MissionConfig(
        name="fraud hybrid BG SHAP n=10k m=100k (XGBoost T=10)",
        dataset=fraud,
        model_wrappers=xgb_models,
        tasks=[bg_task],
        n_values=[10_000],
        m_values=[100_000],
        cache_root=cache_root,
    )))

    # ------------------------------------------------------------------
    # Path-dependent Interactions (m=0) — heavier, skip n=100k
    # ------------------------------------------------------------------
    for label, models in [("medium depth", medium_models), ("high depth", high_models)]:
        for n in [1, 10_000]:
            n_label = {1: "n=1", 10_000: "n=10k"}[n]
            missions.append(Mission(MissionConfig(
                name=f"fraud hybrid PD Interactions {n_label} ({label})",
                dataset=fraud,
                model_wrappers=models,
                tasks=[int_task],
                n_values=[n],
                m_values=[0],
                cache_root=cache_root,
            )))

    missions.append(Mission(MissionConfig(
        name="fraud hybrid PD Interactions n=10k (XGBoost T=10)",
        dataset=fraud,
        model_wrappers=xgb_models,
        tasks=[int_task],
        n_values=[10_000],
        m_values=[0],
        cache_root=cache_root,
    )))

    return missions


def build_experiment() -> Experiment:
    return Experiment(
        name="fraud_hybrid_woodelf_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
