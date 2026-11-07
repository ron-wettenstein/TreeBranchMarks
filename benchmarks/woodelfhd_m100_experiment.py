"""
WoodelfHD Depth Sweep — Background tasks with a 100-row background (m=100).

Same datasets, D sweeps, LightGBM models, per-depth tree counts (T), crash
sentinels and approaches as ``benchmarks/woodelfhd_main_experiment.py``
(the specs and overrides are imported from there, so the two stay in sync),
with these differences:

1. Only the Background tasks run (BG SHAP, BG IV). The path-dependent tasks
   do not use a background dataset, so they would just repeat the main
   experiment.
2. The background dataset is m=100 rows instead of the full training set.
   ControlledMission draws them at random, disjoint from the explained rows
   (n + 100 <= dataset size for every dataset). The explained rows are the
   same as in the main experiment, because the RNG sequence does not change.
3. SHAP runs on all 100 background rows (bg_shap_limit=100), so its times
   are not extrapolated in m. The main experiment ran SHAP on 10 rows and
   extrapolated. Set ``_SHAP_BG_LIMIT = 10`` to match that and cut SHAP
   compute by ~10x.
4. California Housing: the main experiment uses prerecorded notebook times
   for OriginalWoodelf at D=9–15. Those were measured with the full
   background, so here the description says that.

Run from project root:
    python -m benchmarks.woodelfhd_m100_experiment
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from treebranchmarks import Experiment
from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks.core.mission import ApproachDOverride, ControlledMission, PrerecordedTime
from treebranchmarks.core.task import TaskType
from treebranchmarks.datasets import (
    CaliforniaHousingDataset,
    FraudDetectionDataset,
    HIGGSDataset,
    IntrusionDetectionDataset,
)
from treebranchmarks.methods.shap_method import SHAPApproach

from benchmarks import woodelfhd_main_experiment as main

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_ROOT  = main.CACHE_ROOT
RESULTS_DIR = main.RESULTS_DIR

M_BACKGROUND   = 100
_SHAP_BG_LIMIT = 100   # rows actually passed to shap.TreeExplainer (10 = main experiment)

_SHAP = SHAPApproach(bg_shap_limit=_SHAP_BG_LIMIT)

_PRERECORDED_NOTE = "measured with the full training set as background, not m=100"

# ---------------------------------------------------------------------------
# Override adaptation
# ---------------------------------------------------------------------------

def _relabel(spec):
    if isinstance(spec, PrerecordedTime):
        desc = f"{spec.estimation_description}; {_PRERECORDED_NOTE}" if spec.estimation_description else _PRERECORDED_NOTE
        return PrerecordedTime(spec.elapsed_s, estimation_description=desc)
    return spec


def _adapt(overrides: list[ApproachDOverride]) -> list[ApproachDOverride]:
    """Swap in the m=100 SHAP approach and relabel prerecorded times."""
    return [
        ApproachDOverride(
            approach=_SHAP if ov.approach is main._SHAP else ov.approach,
            full_T=ov.full_T,
            model_by_D={D: _relabel(spec) for D, spec in ov.model_by_D.items()},
        )
        for ov in overrides
    ]


# ---------------------------------------------------------------------------
# Mission builder
# ---------------------------------------------------------------------------

def build_missions(cache_root: Path = CACHE_ROOT) -> list[ControlledMission]:
    # (label, dataset, D_values, specs_fn, overrides_fn, n for BG SHAP, n for BG IV)
    datasets = [
        ("fraud_detection",     FraudDetectionDataset(),     main._D_STANDARD,
         main._fraud_specs,   main._fraud_overrides,   118_108,   10_000),
        ("higgs",               HIGGSDataset(),              main._D_STANDARD,
         main._higgs_specs,   main._higgs_overrides,   2_200_000, 10_000),
        ("intrusion_detection", IntrusionDetectionDataset(), main._D_STANDARD,
         main._kdd_specs,     main._kdd_overrides,     2_984_154, 10_000),
        ("california_housing",  CaliforniaHousingDataset(),  main._D_HOUSING,
         main._housing_specs, partial(main._housing_overrides, D_values=main._D_HOUSING), 4_128, 4_128),
    ]

    missions = []
    for label, dataset, D_values, specs_fn, overrides_fn, n_shap, n_iv in datasets:
        specs = specs_fn(cache_root)
        for task_type, sfx, n in [
            (TaskType.BACKGROUND_SHAP,              "BG SHAP", n_shap),
            (TaskType.BACKGROUND_SHAP_INTERACTIONS, "BG IV",   n_iv),
        ]:
            missions.append(main._mission(
                f"{label} {sfx} sweep_D", dataset, D_values,
                _adapt(overrides_fn(task_type, specs)),
                task_type, n, M_BACKGROUND, cache_root,
            ))
    return missions


_SUMMARY_HTML = Path(__file__).parent.parent / "treebranchmarks" / "report" / "benchmark_summaries" / "woodelfhd_m100_summary.html"


def build_experiment() -> Experiment:
    return Experiment(
        name="woodelfhd_m100_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
        summary_html_path=_SUMMARY_HTML,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
