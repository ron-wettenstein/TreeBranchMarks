"""
Woodelf Sparse variants comparison — Background SHAP + Path-Dependent SHAP.

Compares 3 woodelf_sparse variants across Fraud Detection and KDD Intrusion datasets:
  - HybridWoodelfSparseApproach              (woodelf_sparse, default settings)
  - HybridWoodelfSparseNoNeighborTrickApproach (woodelf_sparse, use_neighbor_leaf_trick=False)
  - HybridWoodelfSparseNoFastMNApproach      (woodelf_sparse, use_faster_mn_p2s=False)

Dataset / models
----------------
  Fraud Detection (IEEE-CIS):
    * LightGBM medium (T=10): D = 9, 12, 15, 18
    * LightGBM high   (T=1):  D = 19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 50, 60
    * XGBoost         (T=10): D = 9, 12, 18, 19, 20, 21, 22, 23, 24

  KDD Intrusion Detection:
    * XGBoost (T=10): D = 3, 6, 9, 12   (shallow — m=4M)
    * XGBoost (T=10): D = 15, 18         (deep — m=1M)

Missions
--------
  Background SHAP — Fraud n=10k:
    m=100:  LightGBM medium, LightGBM high (all depths), XGBoost (all depths)
    m=10k:  LightGBM medium, LightGBM high (D=19–24 only), XGBoost (D=9,12,18 only)

  Background SHAP — KDD n=1M:
    m=4M:   XGBoost D = 3, 6, 9, 12
    m=1M:   XGBoost D = 15, 18

  Path-Dependent SHAP — Fraud n=10k:
    LightGBM medium (D=9–18), LightGBM high (D=19–60), XGBoost (D=9–24)

  Path-Dependent SHAP — KDD n=1M:
    XGBoost D = 3, 6, 9, 12
    XGBoost D = 15, 18

Run from the project root:
    python -m benchmarks.woodelf_sparse_variants
"""

from pathlib import Path

from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import FraudDetectionDataset, IntrusionDetectionDataset
from treebranchmarks.models import LightGBMWrapper, XGBoostWrapper
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.methods.woodelf_hybrid_method import (
    HybridWoodelfSparseApproach,
    HybridWoodelfSparseNoNeighborTrickApproach,
    HybridWoodelfSparseNoFastMNApproach,
    HybridWoodelfSparseDirectMNApproach,
)

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

_FRAUD_XGB_BASE = {
    "objective": "reg:squarederror",
    "seed": 123,
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "subsample": 1,
    "colsample_bytree": 1,
}

_KDD_XGB_BASE = {
    "objective": "reg:squarederror",
    "seed": 42,
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "subsample": 1,
    "colsample_bytree": 1,
}

# ---------------------------------------------------------------------------
# Depth lists
# ---------------------------------------------------------------------------

MEDIUM_DEPTHS     = [9, 12, 15, 18]
HIGH_DEPTHS       = [19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 50, 60]
HIGH_DEPTHS_SMALL = [19, 20, 21, 22, 23, 24]   # for m=10k (25–60 too slow)
XGB_DEPTHS        = [9, 12, 18, 19, 20, 21, 22, 23, 24]
XGB_DEPTHS_SMALL  = [9, 12, 18]                # for m=10k

KDD_SHALLOW_DEPTHS = [3, 6, 9, 12]
KDD_DEEP_DEPTHS    = [15, 18]

# ---------------------------------------------------------------------------
# Approaches
# ---------------------------------------------------------------------------

_APPROACHES = [
    HybridWoodelfSparseApproach(),
    HybridWoodelfSparseNoNeighborTrickApproach(),
    HybridWoodelfSparseNoFastMNApproach(),
    HybridWoodelfSparseDirectMNApproach(),
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


def _fraud_xgb_models(depths: list[int], n_estimators: int = 10) -> dict:
    return {
        ModelConfig(
            ensemble_type=EnsembleType.XGBOOST,
            hyperparams={**_FRAUD_XGB_BASE, "max_depth": d, "n_estimators": n_estimators},
            random_state=42,
        ): XGBoostWrapper(task_type="regression")
        for d in depths
    }


def _kdd_xgb_models(depths: list[int], n_estimators: int = 10) -> dict:
    return {
        ModelConfig(
            ensemble_type=EnsembleType.XGBOOST,
            hyperparams={**_KDD_XGB_BASE, "max_depth": d, "n_estimators": n_estimators},
            random_state=42,
        ): XGBoostWrapper(task_type="regression")
        for d in depths
    }

# ---------------------------------------------------------------------------
# Mission builder
# ---------------------------------------------------------------------------

def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    fraud = FraudDetectionDataset(cache_root=cache_root)
    kdd   = IntrusionDetectionDataset(cache_root=cache_root)

    # Fraud models
    medium_models       = _lgbm_models(_LGBM_MEDIUM_BASE, MEDIUM_DEPTHS)
    high_models         = _lgbm_models(_LGBM_HIGH_BASE,   HIGH_DEPTHS)
    high_models_small_m = _lgbm_models(_LGBM_HIGH_BASE,   HIGH_DEPTHS_SMALL)
    fraud_xgb_models    = _fraud_xgb_models(XGB_DEPTHS)
    fraud_xgb_small     = _fraud_xgb_models(XGB_DEPTHS_SMALL)

    # KDD models
    kdd_shallow = _kdd_xgb_models(KDD_SHALLOW_DEPTHS)
    kdd_deep    = _kdd_xgb_models(KDD_DEEP_DEPTHS)

    missions = []

    # ------------------------------------------------------------------
    # Background SHAP — Fraud
    # ------------------------------------------------------------------
    lgbm_by_m = {
        100:    [("medium depth", medium_models), ("high depth", high_models)],
        10_000: [("medium depth", medium_models), ("high depth", high_models_small_m)],
    }
    xgb_by_m = {
        100:    fraud_xgb_models,
        10_000: fraud_xgb_small,
    }

    for m, m_label in [(100, "m=100"), (10_000, "m=10k")]:
        bg_task = Task(TaskType.BACKGROUND_SHAP, _APPROACHES, n_repeats=1, cache_root=cache_root)
        for label, models in lgbm_by_m[m]:
            missions.append(Mission(MissionConfig(
                name=f"fraud BG SHAP n=10k {m_label} ({label})",
                dataset=fraud,
                model_wrappers=models,
                tasks=[bg_task],
                n_values=[10_000],
                m_values=[m],
                cache_root=cache_root,
            )))
        missions.append(Mission(MissionConfig(
            name=f"fraud BG SHAP n=10k {m_label} (XGBoost T=10)",
            dataset=fraud,
            model_wrappers=xgb_by_m[m],
            tasks=[bg_task],
            n_values=[10_000],
            m_values=[m],
            cache_root=cache_root,
        )))

    # ------------------------------------------------------------------
    # Background SHAP — KDD
    # ------------------------------------------------------------------
    bg_task_kdd = Task(TaskType.BACKGROUND_SHAP, _APPROACHES, n_repeats=1, cache_root=cache_root)
    missions.append(Mission(MissionConfig(
        name="kdd intrusion BG SHAP n=1M m=4M (D=3–12)",
        dataset=kdd,
        model_wrappers=kdd_shallow,
        tasks=[bg_task_kdd],
        n_values=[1_000_000],
        m_values=[4_000_000],
        cache_root=cache_root,
    )))
    missions.append(Mission(MissionConfig(
        name="kdd intrusion BG SHAP n=1M m=1M (D=15–18)",
        dataset=kdd,
        model_wrappers=kdd_deep,
        tasks=[bg_task_kdd],
        n_values=[1_000_000],
        m_values=[1_000_000],
        cache_root=cache_root,
    )))

    # ------------------------------------------------------------------
    # Path-Dependent SHAP — Fraud
    # ------------------------------------------------------------------
    pd_task_fraud = Task(TaskType.PATH_DEPENDENT_SHAP, _APPROACHES, n_repeats=1, cache_root=cache_root)
    for label, models in [("medium depth", medium_models), ("high depth", high_models)]:
        missions.append(Mission(MissionConfig(
            name=f"fraud PD SHAP n=10k ({label})",
            dataset=fraud,
            model_wrappers=models,
            tasks=[pd_task_fraud],
            n_values=[10_000],
            m_values=[0],
            cache_root=cache_root,
        )))
    missions.append(Mission(MissionConfig(
        name="fraud PD SHAP n=10k (XGBoost T=10)",
        dataset=fraud,
        model_wrappers=fraud_xgb_models,
        tasks=[pd_task_fraud],
        n_values=[10_000],
        m_values=[0],
        cache_root=cache_root,
    )))

    # ------------------------------------------------------------------
    # Path-Dependent SHAP — KDD
    # ------------------------------------------------------------------
    pd_task_kdd = Task(TaskType.PATH_DEPENDENT_SHAP, _APPROACHES, n_repeats=1, cache_root=cache_root)
    missions.append(Mission(MissionConfig(
        name="kdd intrusion PD SHAP n=1M (D=3–12)",
        dataset=kdd,
        model_wrappers=kdd_shallow,
        tasks=[pd_task_kdd],
        n_values=[1_000_000],
        m_values=[0],
        cache_root=cache_root,
    )))
    missions.append(Mission(MissionConfig(
        name="kdd intrusion PD SHAP n=1M (D=15–18)",
        dataset=kdd,
        model_wrappers=kdd_deep,
        tasks=[pd_task_kdd],
        n_values=[1_000_000],
        m_values=[0],
        cache_root=cache_root,
    )))

    return missions


def build_experiment() -> Experiment:
    return Experiment(
        name="woodelf_sparse_variants",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
