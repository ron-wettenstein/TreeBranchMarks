"""
SHAP vs TreeGrad — HIGGS, GradientBoosting depth sweep.

Compares path-dependent SHAP implementations that support sklearn models:
  - shap package (reference)
  - TreeGrad (path-dependent SHAP, sklearn models only)

Dataset / models
----------------
  * HIGGS dataset (binary classification, 25 000 training samples)
  * GradientBoostingClassifier (T=1): D = 9, 12, 18, 21, 24

Missions
--------
For n in [1, 1k, 10k, 100k] — one mission per n, all depths in each mission.

Run from the project root:
    python -m benchmarks.fraud_shap_vs_treegrad_experiment
"""

from pathlib import Path

from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import HIGGSDataset
from treebranchmarks.models import GradientBoostingWrapper
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.treegrad_method import TreeGradApproach

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

DEPTHS       = [9, 12, 18, 21, 24]
N_ESTIMATORS = 1

N_TRAIN = 25_000

_GB_BASE = {
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.1,
    "max_train_samples": N_TRAIN, # Builtin hyperparam of the treebranchmarks framework.
}

_APPROACHES = [SHAPApproach(), TreeGradApproach()]


def _gb_models(depths: list[int]) -> dict:
    return {
        ModelConfig(
            ensemble_type=EnsembleType.GRADIENT_BOOSTING,
            hyperparams={**_GB_BASE, "max_depth": d},
            random_state=42,
        ): GradientBoostingWrapper(task_type="classification")
        for d in depths
    }


def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    higgs  = HIGGSDataset(cache_root=cache_root)
    models = _gb_models(DEPTHS)
    task   = Task(TaskType.PATH_DEPENDENT_SHAP, _APPROACHES, n_repeats=1, cache_root=cache_root)

    missions = []
    for n in [1, 1_000, 10_000, 100_000]:
        n_label = {1: "n=1", 1_000: "n=1k", 10_000: "n=10k", 100_000: "n=100k"}[n]
        missions.append(Mission(MissionConfig(
            name=f"higgs SHAP vs TreeGrad {n_label} (GB T={N_ESTIMATORS})",
            dataset=higgs,
            model_wrappers=models,
            tasks=[task],
            n_values=[n],
            m_values=[0],
            cache_root=cache_root,
        )))

    return missions


def build_experiment() -> Experiment:
    return Experiment(
        name="higgs_shap_vs_treegrad_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
