"""
sklearn datasets experiment: SHAP vs WoodelfHD on PD SHAP and BG SHAP.

Compares shap (TreeExplainer) against WoodelfHD across seven datasets and four
tree depths, matching the QuadratureSHAP benchmark suite workloads.

Datasets
--------
  breast_cancer      — sklearn binary classification (569 rows,   30 features)
  diabetes           — sklearn regression            (442 rows,   10 features)
  digits             — sklearn multiclass            (1797 rows,  64 features, 10 classes)
  california_housing — sklearn regression            (20640 rows,  8 features)
  easy_linear        — synthetic binary, linear boundary (40k rows, 50 features)
  random_labels      — synthetic binary, random labels   (40k rows, 50 features)
  random_regression  — synthetic regression, random targets (40k rows, 50 features)

Models
------
  XGBoost, D ∈ {4, 8, 16, 30}
  n_estimators matched to reference benchmark rounds per dataset.

Tasks
-----
  Path-Dependent SHAP   (n per dataset, m=0)
  Background SHAP       (n per dataset, m=100)

Run from project root:
    python -m benchmarks.sklearn_datasets_experiment
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.datasets import (
    BreastCancerDataset,
    DiabetesDataset,
    DigitsDataset,
    CaliforniaHousingDataset,
    EasyLinearDataset,
    RandomLabelsDataset,
    RandomRegressionDataset,
)
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.woodelf_original_and_hd_method import WoodelfHDApproach
from treebranchmarks.models.xgboost_model import XGBoostWrapper

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

D_VALUES = [4, 8, 16, 30]

_XGB_BASE = dict(
    tree_method="hist",
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_weight=0.0,
)


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

@dataclass
class _DatasetConfig:
    name: str
    dataset: object
    task_type: str           # "classification" or "regression"
    n_estimators: int        # rounds, matching the reference benchmark
    n_explain: int           # rows to explain (n_values)


_DATASET_CONFIGS: list[_DatasetConfig] = [
    _DatasetConfig("breast_cancer",      BreastCancerDataset(),      "classification", 200,  100),
    _DatasetConfig("diabetes",           DiabetesDataset(),           "regression",     300,  100),
    _DatasetConfig("digits",             DigitsDataset(),             "classification", 250,  400),
    _DatasetConfig("california_housing", CaliforniaHousingDataset(),  "regression",     300,  512),
    _DatasetConfig("easy_linear",        EasyLinearDataset(),         "classification", 200,  512),
    _DatasetConfig("random_labels",      RandomLabelsDataset(),       "classification", 200,  512),
    _DatasetConfig("random_regression",  RandomRegressionDataset(),   "regression",     200,  512),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xgb_models(task_type: str, n_estimators: int, cache_root: Path) -> dict:
    """One XGBoostWrapper per depth."""
    return {
        ModelConfig(
            ensemble_type=EnsembleType.XGBOOST,
            hyperparams={**_XGB_BASE, "max_depth": d, "n_estimators": n_estimators},
            random_state=42,
        ): XGBoostWrapper(task_type=task_type)
        for d in D_VALUES
    }


def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    _shap    = SHAPApproach()
    _woodelf = WoodelfHDApproach()

    missions = []
    for cfg in _DATASET_CONFIGS:
        models = _xgb_models(cfg.task_type, cfg.n_estimators, cache_root)

        pd_task = Task(TaskType.PATH_DEPENDENT_SHAP, [_shap, _woodelf],
                       n_repeats=1, cache_root=cache_root)
        bg_task = Task(TaskType.BACKGROUND_SHAP,     [_shap, _woodelf],
                       n_repeats=1, cache_root=cache_root)

        missions.append(Mission(MissionConfig(
            name=f"{cfg.name}: PD SHAP (D-sweep)",
            dataset=cfg.dataset,
            model_wrappers=models,
            tasks=[pd_task],
            n_values=[cfg.n_explain],
            m_values=[0],
            cache_root=cache_root,
        )))
        missions.append(Mission(MissionConfig(
            name=f"{cfg.name}: BG SHAP m=100 (D-sweep)",
            dataset=cfg.dataset,
            model_wrappers=models,
            tasks=[bg_task],
            n_values=[cfg.n_explain],
            m_values=[100],
            cache_root=cache_root,
        )))

    return missions


def build_experiment() -> Experiment:
    return Experiment(
        name="sklearn_datasets_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
