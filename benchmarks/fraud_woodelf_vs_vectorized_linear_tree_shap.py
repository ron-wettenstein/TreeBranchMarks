"""
Woodelf vs VectorizedLinearTreeSHAP — Fraud Detection depth sweep.

Compares two path-dependent SHAP implementations across tree depths 1–11:
  - Woodelf (WoodelfExplainer, tree_path_dependent)
  - VectorizedLinearTreeSHAP (vectorized_linear_tree_shap, NLT on, default p2m)

Dataset / models
----------------
  * IEEE-CIS Fraud Detection dataset (binary classification)
  * XGBoost (T=10): D = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

Missions
--------
One mission per n value — each is a D sweep across all 11 depths:
  - PD SHAP  n=1,       m=0
  - PD SHAP  n=100,     m=0
  - PD SHAP  n=1 000,   m=0
  - PD SHAP  n=10 000,  m=0
  - PD SHAP  n=100 000, m=0
  - PD SHAP  n=400 000, m=0

Run from the project root:
    python -m benchmarks.fraud_woodelf_vs_vectorized_linear_tree_shap
"""

from pathlib import Path

from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import FraudDetectionDataset
from treebranchmarks.models import XGBoostWrapper
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.methods.woodelf_method import WoodelfApproach
from treebranchmarks.methods.linear_tree_shap_method import VectorizedLinearTreeSHAPApproach

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

_XGB_BASE = {
    "objective": "reg:squarederror",
    "seed": 123,
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "subsample": 1,
    "colsample_bytree": 1,
    "n_estimators": 10,
}

DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

_APPROACHES = [WoodelfApproach(), VectorizedLinearTreeSHAPApproach()]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xgb_models(depths: list[int]) -> dict:
    return {
        ModelConfig(
            ensemble_type=EnsembleType.XGBOOST,
            hyperparams={**_XGB_BASE, "max_depth": d},
            random_state=42,
        ): XGBoostWrapper(task_type="regression")
        for d in depths
    }


# ---------------------------------------------------------------------------
# Mission builder
# ---------------------------------------------------------------------------

def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    fraud = FraudDetectionDataset(cache_root=cache_root)
    models = _xgb_models(DEPTHS)

    task = Task(TaskType.PATH_DEPENDENT_SHAP, _APPROACHES, n_repeats=1, cache_root=cache_root)

    n_labels = {
        1:        "n=1",
        100:      "n=100",
        1_000:    "n=1k",
        10_000:   "n=10k",
        100_000:  "n=100k",
        400_000:  "n=400k",
    }

    missions = []
    for n, label in n_labels.items():
        missions.append(Mission(MissionConfig(
            name=f"fraud woodelf vs vec LTSHAP {label} (XGBoost T=10)",
            dataset=fraud,
            model_wrappers=models,
            tasks=[task],
            n_values=[n],
            m_values=[0],
            cache_root=cache_root,
        )))

    return missions


def build_experiment() -> Experiment:
    return Experiment(
        name="fraud_woodelf_vs_vectorized_linear_tree_shap",
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
