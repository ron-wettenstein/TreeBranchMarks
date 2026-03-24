"""
Fraud Detection depth-sweep experiment: compare WoodelfHD vs OriginalWoodelf (CPU only).

Dataset / models
----------------
* IEEE-CIS Fraud Detection dataset (binary classification)
* LightGBM — one model per depth D ∈ {1,2,3,4,5,6,7,8,9,10}, T=10 trees

Missions (2 total — one per task type)
---------------------------------------
  1. BG SHAP    n=100 000, m=400 000, D-sweep
  2. BG SHAP IV n=10 000,  m=400 000, D-sweep

Run from the project root:
    python -m benchmarks.fraud_woodelfhd_vs_original_depth_experiment
"""

from pathlib import Path

from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import FraudDetectionDataset
from treebranchmarks.models import LightGBMWrapper
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.methods.woodelf_original_and_hd_method import (
    WoodelfHDApproach,
    OriginalWoodelfApproach,
)

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

D_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

_LGBM_BASE = {
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


def _lgbm_models(depths: list[int], cache_root: Path) -> dict:
    """Build a {ModelConfig: LightGBMWrapper} dict, one entry per depth."""
    return {
        ModelConfig(
            ensemble_type=EnsembleType.LIGHTGBM,
            hyperparams={**_LGBM_BASE, "max_depth": d},
            random_state=42,
        ): LightGBMWrapper(task_type="regression")
        for d in depths
    }


def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    approaches = [WoodelfHDApproach(), OriginalWoodelfApproach()]

    fraud = FraudDetectionDataset(cache_root=cache_root)
    models = _lgbm_models(D_VALUES, cache_root)

    bg_shap_task = Task(TaskType.BACKGROUND_SHAP,              approaches, n_repeats=1, cache_root=cache_root)
    bg_iv_task   = Task(TaskType.BACKGROUND_SHAP_INTERACTIONS, approaches, n_repeats=1, cache_root=cache_root)

    return [
        Mission(MissionConfig(
            name="fraud: BG SHAP n=100k m=400k (D-sweep)",
            dataset=fraud,
            model_wrappers=models,
            tasks=[bg_shap_task],
            n_values=[100_000],
            m_values=[400_000],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="fraud: BG SHAP IV n=10k m=400k (D-sweep)",
            dataset=fraud,
            model_wrappers=models,
            tasks=[bg_iv_task],
            n_values=[10_000],
            m_values=[400_000],
            cache_root=cache_root,
        )),
    ]


def build_experiment() -> Experiment:
    return Experiment(
        name="fraud_woodelfhd_vs_original_depth_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)