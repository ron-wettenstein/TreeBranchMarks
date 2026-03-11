"""
Base Woodelf progress experiment.

Combines all missions from four experiments to compare the full historical
Woodelf algorithm family against the shap reference:
  - shap (SHAPApproach)
  - Woodelf (WoodelfApproach)
  - Woodelf ECAI (WoodelfECAIApproach)
  - Woodelf AAAI (WoodelfAAAIApproach)
  - WoodelfHD (WoodelfHDApproach)

Sources
-------
  fraud_detection_experiment     — XGBoost D=6, T=10 on Fraud Detection
  fraud_depth_experiment         — LightGBM medium/high depth sweep on Fraud Detection
  small_dataset_experiment       — XGBoost D=3 T=10 (California Housing) + DecisionTree D=7 (Breast Cancer)
  intrusion_detection_experiment — HistGradientBoosting D=6, T=10 on Intrusion Detection

Run from the project root:
    python -m benchmarks.base_woodelf_progress_experiment
"""

from pathlib import Path

from treebranchmarks import Experiment, Mission
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.woodelf_method import WoodelfApproach
from treebranchmarks.methods.woodelf_historical_methods import (
    WoodelfECAIApproach,
    WoodelfAAAIApproach,
    WoodelfHDApproach,
)
from benchmarks.fraud_detection_experiment import build_missions as fraud_detection_missions
from benchmarks.fraud_depth_experiment import build_missions as fraud_depth_missions
from benchmarks.small_dataset_experiment import build_missions as small_dataset_missions
from benchmarks.intrusion_detection_experiment import build_missions as intrusion_detection_missions

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")

_APPROACHES = [
    SHAPApproach(),
    WoodelfApproach(),
    WoodelfECAIApproach(),
    WoodelfAAAIApproach(),
    WoodelfHDApproach(),
]


def build_missions(cache_root: Path = CACHE_ROOT) -> list[Mission]:
    return (
        fraud_detection_missions(cache_root=cache_root, approaches=_APPROACHES)
        + fraud_depth_missions(cache_root=cache_root, approaches=_APPROACHES)
        + small_dataset_missions(cache_root=cache_root, approaches=_APPROACHES)
        + intrusion_detection_missions(cache_root=cache_root, approaches=_APPROACHES)
    )


def build_experiment() -> Experiment:
    return Experiment(
        name="base_woodelf_progress_experiment",
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
