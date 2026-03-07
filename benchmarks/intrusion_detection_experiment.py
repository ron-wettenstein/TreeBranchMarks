"""
Intrusion Detection experiment: compare Woodelf vs shap on the KDD Cup 1999 dataset.

Dataset / model
---------------
* KDD Cup 1999 Intrusion Detection + HistGradientBoostingRegressor (max_depth=6, max_iter=10)

Missions
--------
1. Background SHAP    (sweep n)  — m=5000000, n=[1, 10000, 100000, 5000000]
2. Path-Dependent SHAP           — m=0,       n=[1000000]
3. Background SHAP IV            — m=5000000, n=[10000]
4. Path-Dependent SHAP IV        — m=0,       n=[10000]

Run from the project root:
    python -m benchmarks.intrusion_detection_experiment
"""

from pathlib import Path

from treebranchmarks import Experiment, Mission, MissionConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import IntrusionDetectionDataset
from treebranchmarks.models import HistGradientBoostingWrapper
from treebranchmarks.tasks import (
    PathDependentSHAPTask,
    BackgroundSHAPTask,
    BackgroundSHAPInteractionsTask,
    PathDependentInteractionsTask,
)

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

HGB_PARAMS = {
    "max_iter": 10,
    "max_depth": 6,
    "max_leaf_nodes": None,
}


def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    """
    Return all missions for this experiment.

    Follows the build_missions() convention so a combined experiment can do:
        from benchmarks.intrusion_detection_experiment import build_missions as intrusion_missions
        Experiment(name="combined", missions=intrusion_missions() + other_missions(), ...)
    """
    intrusion = IntrusionDetectionDataset(cache_root=cache_root)

    hgb_d6_t10 = {
        ModelConfig(
            ensemble_type=EnsembleType.HIST_GRADIENT_BOOSTING,
            hyperparams=HGB_PARAMS,
            random_state=42,
        ): HistGradientBoostingWrapper(task_type="regression")
    }

    pd_shap_task = PathDependentSHAPTask(n_repeats=1, cache_root=cache_root)
    pd_iv_task   = PathDependentInteractionsTask(n_repeats=1, cache_root=cache_root)
    bg_shap_task = BackgroundSHAPTask(n_repeats=1, cache_root=cache_root)
    bg_iv_task   = BackgroundSHAPInteractionsTask(n_repeats=1, cache_root=cache_root)

    return [
        Mission(MissionConfig(
            name="intrusion: BG SHAP (sweep n)",
            dataset=intrusion,
            model_wrappers=hgb_d6_t10,
            tasks=[bg_shap_task],
            n_values=[1, 10_000, 100_000, 4_000_000],
            m_values=[4_000_000],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="intrusion: PD SHAP",
            dataset=intrusion,
            model_wrappers=hgb_d6_t10,
            tasks=[pd_shap_task],
            n_values=[1_000_000],
            m_values=[0],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="intrusion: BG SHAP IV",
            dataset=intrusion,
            model_wrappers=hgb_d6_t10,
            tasks=[bg_iv_task],
            n_values=[10_000],
            m_values=[4_000_000],
            cache_root=cache_root,
        )),
        Mission(MissionConfig(
            name="intrusion: PD SHAP IV",
            dataset=intrusion,
            model_wrappers=hgb_d6_t10,
            tasks=[pd_iv_task],
            n_values=[10_000],
            m_values=[0],
            cache_root=cache_root,
        )),
    ]


def build_experiment() -> Experiment:
    return Experiment(
        name="intrusion_detection_experiment",
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
