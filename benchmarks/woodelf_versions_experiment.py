"""
Woodelf versions experiment.

Compares specific woodelf releases head-to-head across the full
base_woodelf_progress_experiment mission suite:

  - SHAP            (SHAPApproach)
  - Woodelf v0.4.2  (WoodelfVersionApproach "v0.4.2")
  - Woodelf v0.4.3  (WoodelfVersionApproach "v0.4.3")
  - Woodelf feature/single_pattern_generation

Sources (same as base_woodelf_progress_experiment)
---------------------------------------------------
  fraud_detection_experiment
  fraud_depth_experiment
  small_dataset_experiment
  intrusion_detection_experiment

Run from the project root:
    python -m benchmarks.woodelf_versions_experiment
"""

from pathlib import Path

from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks import Experiment
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.woodelf_version_method import WoodelfVersionApproach
from benchmarks.fraud_detection_experiment import build_missions as fraud_detection_missions
from benchmarks.fraud_depth_experiment import build_missions as fraud_depth_missions
from benchmarks.small_dataset_experiment import build_missions as small_dataset_missions
from benchmarks.intrusion_detection_experiment import build_missions as intrusion_detection_missions

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

_APPROACHES = [
    SHAPApproach(),
    WoodelfVersionApproach("v0.4.2",                        cache_root=CACHE_ROOT),
    WoodelfVersionApproach("v0.4.3",                        cache_root=CACHE_ROOT),
    WoodelfVersionApproach("feature/single_pattern_generation", cache_root=CACHE_ROOT),
    WoodelfVersionApproach("feature/explainer_use_sparse_on_small_data", cache_root=CACHE_ROOT),
]


def build_missions(cache_root: Path = CACHE_ROOT):
    return (
        fraud_detection_missions(cache_root=cache_root, approaches=_APPROACHES)
        + fraud_depth_missions(cache_root=cache_root, approaches=_APPROACHES)
        + small_dataset_missions(cache_root=cache_root, approaches=_APPROACHES)
        + intrusion_detection_missions(cache_root=cache_root, approaches=_APPROACHES)
    )


def build_experiment() -> Experiment:
    return Experiment(
        name="woodelf_versions_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
