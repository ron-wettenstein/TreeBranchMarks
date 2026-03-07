"""
Fraud Detection experiment: compare Woodelf vs shap on the IEEE-CIS dataset.

Dataset / model
---------------
* IEEE-CIS Fraud Detection + XGBoost (max_depth=6, T=10)  — binary classification

Missions
--------
1. Path-Dependent SHAP          — sweep n=[10000, 100000],         m=0
2. Path-Dependent SHAP IV       — sweep n=[10000, 100000],         m=0
3. Background SHAP              — n=100000, sweep m=[100, 1000, 10000, 100000, 500000]
4. Background SHAP IV           — n=10000,  sweep m=[100, 1000, 10000, 100000, 500000]

Run from the project root:
    python -m benchmarks.fraud_detection_experiment
"""

from pathlib import Path

from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import FraudDetectionDataset
from treebranchmarks.models import XGBoostWrapper
from treebranchmarks.tasks import (
    PathDependentSHAPTask,
    BackgroundSHAPTask,
    BackgroundSHAPInteractionsTask,
    PathDependentInteractionsTask,
)

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

XGB_PARAMS = {
    "n_estimators": 10,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1,
    "colsample_bytree": 0.8,
    "nthread": 1,
}


def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    """
    Return all missions for this experiment.

    Follows the build_missions() convention so a combined experiment can do:
        from benchmarks.fraud_detection_experiment import build_missions as fraud_missions
        Experiment(name="combined", missions=fraud_missions() + other_missions(), ...)
    """
    fraud = FraudDetectionDataset(cache_root=cache_root)

    xgb_d6_t10 = {
        ModelConfig(
            ensemble_type=EnsembleType.XGBOOST,
            hyperparams=XGB_PARAMS,
            random_state=123,
        ): XGBoostWrapper(task_type="regression")
    }

    pd_shap_task = PathDependentSHAPTask(n_repeats=1, cache_root=cache_root)
    pd_iv_task   = PathDependentInteractionsTask(n_repeats=1, cache_root=cache_root)
    bg_shap_task = BackgroundSHAPTask(n_repeats=1, cache_root=cache_root)
    bg_iv_task   = BackgroundSHAPInteractionsTask(n_repeats=1, cache_root=cache_root)

    return [
        Mission(MissionConfig(
            name="fraud: PD SHAP (sweep n)",
            dataset=fraud,
            model_wrappers=xgb_d6_t10,
            tasks=[pd_shap_task],
            n_values=[10_000, 100_000, 500_000],
            m_values=[0],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="fraud: PD SHAP IV (sweep n)",
            dataset=fraud,
            model_wrappers=xgb_d6_t10,
            tasks=[pd_iv_task],
            n_values=[100, 1000, 10_000],
            m_values=[0],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="fraud: BG SHAP n=100000 (sweep m)",
            dataset=fraud,
            model_wrappers=xgb_d6_t10,
            tasks=[bg_shap_task],
            n_values=[100_000],
            m_values=[100, 1_000, 10_000, 100_000, 500_000],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="fraud: BG SHAP IV n=10000 (sweep m)",
            dataset=fraud,
            model_wrappers=xgb_d6_t10,
            tasks=[bg_iv_task],
            n_values=[10_000],
            m_values=[100, 1_000, 10_000, 100_000, 500_000],
            cache_root=cache_root,
        )),
    ]


def build_experiment() -> Experiment:
    return Experiment(
        name="fraud_detection_experiment",
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
