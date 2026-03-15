"""
Example experiment: compare path-dependent vs interventional SHAP on
California Housing.  Each mission has exactly ONE free variable.

Run from the project root:
    python -m benchmarks.example_experiment

Results are cached under cache/ and results/ (both gitignored).
The HTML report is written to results/shap_comparison_v1.html.
"""

from pathlib import Path

from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks import (
    Experiment,
    Mission,
    MissionConfig,
)
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.datasets import CaliforniaHousingDataset
from treebranchmarks.models import LightGBMWrapper
from treebranchmarks.core.task import Task, TaskType
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.woodelf_explainer_method import WoodelfApproach

CACHE_ROOT = Path("cache")
RESULTS_DIR = Path("results")

# Constants shared across missions
FIXED_DEPTH  = 6
FIXED_N      = 200
FIXED_M      = 100
N_ESTIMATORS = 10 # 100


def _lgbm(depth: int) -> dict[ModelConfig, LightGBMWrapper]:
    """Single LightGBM model config at the given depth."""
    return {
        ModelConfig(
            ensemble_type=EnsembleType.LIGHTGBM,
            hyperparams={"n_estimators": N_ESTIMATORS, "max_depth": depth},
        ): LightGBMWrapper(task_type="regression")
    }


def build_experiment() -> Experiment:
    california = CaliforniaHousingDataset(cache_root=CACHE_ROOT)

    _approaches = [SHAPApproach(), WoodelfApproach()]
    path_dep_task   = Task(TaskType.PATH_DEPENDENT_SHAP, _approaches, n_repeats=1, cache_root=CACHE_ROOT)
    background_task = Task(TaskType.BACKGROUND_SHAP,     _approaches, n_repeats=1, cache_root=CACHE_ROOT)

    # ------------------------------------------------------------------
    # Path-dependent SHAP — sweep n  (D fixed)
    # ------------------------------------------------------------------
    pd_sweep_n = Mission(MissionConfig(
        name="path-dep: sweep n",
        dataset=california,
        model_wrappers=_lgbm(FIXED_DEPTH),
        tasks=[path_dep_task],
        n_values=[1, 10, 100, 1000, 10000],
        m_values=[0],
        cache_root=CACHE_ROOT,
    ))

    # ------------------------------------------------------------------
    # Path-dependent SHAP — sweep D  (n fixed)
    # ------------------------------------------------------------------
    pd_sweep_D = Mission(MissionConfig(
        name="path-dep: sweep D",
        dataset=california,
        model_wrappers={**_lgbm(3), **_lgbm(6), **_lgbm(9), **_lgbm(12)},
        tasks=[path_dep_task],
        n_values=[FIXED_N],
        m_values=[0],
        cache_root=CACHE_ROOT,
    ))

    # ------------------------------------------------------------------
    # Background (interventional) SHAP — sweep n  (m, D fixed)
    # ------------------------------------------------------------------
    bg_sweep_n = Mission(MissionConfig(
        name="background: sweep n",
        dataset=california,
        model_wrappers=_lgbm(FIXED_DEPTH),
        tasks=[background_task],
        n_values=[1, 10, 100, 1000, 10000],
        m_values=[FIXED_M],
        cache_root=CACHE_ROOT,
    ))

    # ------------------------------------------------------------------
    # Background (interventional) SHAP — sweep m  (n, D fixed)
    # ------------------------------------------------------------------
    bg_sweep_m = Mission(MissionConfig(
        name="background: sweep m",
        dataset=california,
        model_wrappers=_lgbm(FIXED_DEPTH),
        tasks=[background_task],
        n_values=[FIXED_N],
        m_values=[1, 10, 100, 1000, 10000],
        cache_root=CACHE_ROOT,
    ))

    # # ------------------------------------------------------------------
    # # Background (interventional) SHAP — sweep D  (n, m fixed)
    # # ------------------------------------------------------------------
    # bg_sweep_D = Mission(MissionConfig(
    #     name="background: sweep D",
    #     dataset=california,
    #     model_wrappers={**_lgbm(3), **_lgbm(6), **_lgbm(9), **_lgbm(12)},
    #     tasks=[background_task],
    #     n_values=[FIXED_N],
    #     m_values=[FIXED_M],
    #     cache_root=CACHE_ROOT,
    # ))

    return Experiment(
        name="shap_comparison_v1",
        missions=[pd_sweep_n, pd_sweep_D, bg_sweep_n, bg_sweep_m], #bg_sweep_D],
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=True,
        delete_model_cache=True,
        delete_results=True,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
