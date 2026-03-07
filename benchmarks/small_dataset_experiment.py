"""
Small-dataset experiment: compare Woodelf vs shap across four task types.

Datasets / models
-----------------
* California Housing  + XGBoost  (D=3, T=10)  — regression
* Breast Cancer       + DecisionTree (D=10, T=1) — classification

Missions
--------
California Housing / XGBoost:
  1. Path-Dependent SHAP          — sweep n=[1, 100, 1000],  m=0
  2. Path-Dependent SHAP IV       — sweep n=[1, 100, 1000],  m=0
  3. Background SHAP              — sweep m=[100, 1000, 10000], n=1000
  4. Background SHAP IV           — n=1000, m=10000  (single point)

Breast Cancer / DecisionTree:
  5. Path-Dependent SHAP          — sweep n=[1, 100],  m=0
  6. Path-Dependent SHAP IV       — sweep n=[1, 100],  m=0

Run from the project root:
    python -m benchmarks.small_dataset_experiment
"""

from pathlib import Path

from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import CaliforniaHousingDataset, BreastCancerDataset
from treebranchmarks.models import XGBoostWrapper, DecisionTreeWrapper
from treebranchmarks.tasks import (
    PathDependentSHAPTask,
    BackgroundSHAPTask,
    BackgroundSHAPInteractionsTask,
    PathDependentInteractionsTask,
)

CACHE_ROOT   = Path("cache")
RESULTS_DIR  = Path("results")


def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    """
    Return all missions for this experiment.

    Keeping missions in a standalone function lets a combined experiment
    import and concatenate missions from multiple files:

        from benchmarks.small_dataset_experiment import build_missions as small_missions
        from benchmarks.other_experiment       import build_missions as other_missions

        Experiment(name="combined", missions=small_missions() + other_missions(), ...)
    """
    california    = CaliforniaHousingDataset(cache_root=cache_root, use_cache=False)
    breast_cancer = BreastCancerDataset(cache_root=cache_root, use_cache=False)

    # Models
    xgb_d3_t10 = {
        ModelConfig(
            ensemble_type=EnsembleType.XGBOOST,
            hyperparams={"n_estimators": 10, "max_depth": 3},
        ): XGBoostWrapper(task_type="regression", use_cache=False)
    }
    dt_d10 = {
        ModelConfig(
            ensemble_type=EnsembleType.DECISION_TREE,
            hyperparams={"max_depth": 10},
        ): DecisionTreeWrapper(task_type="classification", use_cache=False)
    }

    # Tasks
    pd_shap_task = PathDependentSHAPTask(n_repeats=1, cache_root=cache_root)
    pd_iv_task   = PathDependentInteractionsTask(n_repeats=1, cache_root=cache_root)
    bg_shap_task = BackgroundSHAPTask(n_repeats=1, cache_root=cache_root)
    bg_iv_task   = BackgroundSHAPInteractionsTask(n_repeats=1, cache_root=cache_root)

    return [
        # California Housing / XGBoost
        Mission(MissionConfig(
            name="housing: PD SHAP (sweep n)",
            dataset=california,
            model_wrappers=xgb_d3_t10,
            tasks=[pd_shap_task],
            n_values=[1, 100, 1000],
            m_values=[0],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="housing: PD SHAP IV (sweep n)",
            dataset=california,
            model_wrappers=xgb_d3_t10,
            tasks=[pd_iv_task],
            n_values=[1, 100, 1000],
            m_values=[0],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="housing: BG SHAP n=1000 (sweep m)",
            dataset=california,
            model_wrappers=xgb_d3_t10,
            tasks=[bg_shap_task],
            n_values=[1000],
            m_values=[100, 1000, 10000],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="housing: BG SHAP IV n=1000, m=10000",
            dataset=california,
            model_wrappers=xgb_d3_t10,
            tasks=[bg_iv_task],
            n_values=[1000],
            m_values=[10000],
            cache_root=cache_root,
        )),
        # Breast Cancer / DecisionTree
        Mission(MissionConfig(
            name="cancer: PD SHAP (sweep n)",
            dataset=breast_cancer,
            model_wrappers=dt_d10,
            tasks=[pd_shap_task],
            n_values=[1, 100],
            m_values=[0],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="cancer: PD SHAP IV (sweep n)",
            dataset=breast_cancer,
            model_wrappers=dt_d10,
            tasks=[pd_iv_task],
            n_values=[1, 100],
            m_values=[0],
            cache_root=cache_root,
        )),
    ]


def build_experiment() -> Experiment:
    return Experiment(
        name="small_dataset_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=True,
        delete_model_cache=True,
        delete_results=True,
    )


if __name__ == "__main__":
    experiment = build_experiment()
    experiment.run()
    report_path = experiment.generate_html()
    print(f"\nOpen the report in your browser:\n  {report_path.resolve()}")
