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
   background, so here the description says that. BG IV D=9 is run here
   (see 5). The others would take a day or more. Nearly all of their time
   (97–98% in the source notebook) is building M matrices, which does not
   depend on the background, so m=100 would give almost the same times.
5. Some cells run on more trees than in the main experiment (_MORE_TREES).
   Cells whose reduced-tree estimate in an earlier m=100 run was cheap enough
   run on all 100 trees. On California Housing, PLTreeSHAP also runs the
   depths where the main experiment gave it OriginalWoodelf's prerecorded
   times (the two share one per-depth table there).

Run from project root:
    python -m benchmarks.woodelfhd_m100_experiment
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

from treebranchmarks import Experiment
from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks.core.mission import MEMORY_CRASH, ApproachDOverride, ControlledMission, ModelSpec, PrerecordedTime
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


# Cells that run on more trees than in the main experiment: approach -> (dataset, task) -> {D: T}.
# These cells were estimated from 1 or 10 trees in an earlier m=100 run and are cheap enough to
# measure on all 100. When a cell is added here, its old estimate must be removed from the
# method's results JSON (e.g. woodelf_hd.json), or it is served from the cache.
# California Housing PLTreeSHAP D=12-15 BG SHAP and D=9-12 BG IV, and OriginalWoodelf BG IV D=9:
# the main experiment gives these OriginalWoodelf's prerecorded times (PLTreeSHAP shares its
# per-depth table there), so here they are run. MEMORY_CRASH marks a cell that crashed on all
# 100 trees: PLTreeSHAP BG SHAP D=15 there, because pltreeshap keeps 2^depth copies of every node
# for all trees at once (~73 GB for that model), more than a Colab High-RAM runtime has.
_BG_SHAP, _BG_IV = TaskType.BACKGROUND_SHAP, TaskType.BACKGROUND_SHAP_INTERACTIONS
_MORE_TREES = {
    main._PLTREESHAP_FTS: {
        ("fraud_detection",     _BG_SHAP): {12: 100, 15: 100},
        ("fraud_detection",     _BG_IV):   {12: 100},
        ("higgs",               _BG_IV):   {9: 100},
        ("intrusion_detection", _BG_SHAP): {9: 100, 12: 100, 15: 100},
        ("intrusion_detection", _BG_IV):   {12: 100},
        ("california_housing",  _BG_SHAP): {9: 100, 12: 100, 15: MEMORY_CRASH},
        ("california_housing",  _BG_IV):   {9: 100, 12: 100},
    },
    main._WOODELF_HD: {
        ("higgs",               _BG_IV):   {9: 100, 12: 100, 15: 100, 18: 100},
        ("intrusion_detection", _BG_SHAP): {12: 100},
        ("intrusion_detection", _BG_IV):   {12: 100, 15: 100, 18: 100},
    },
    main._WOODELF_AAAI: {
        ("intrusion_detection", _BG_SHAP): {9: 100, 12: 100},
        ("california_housing",  _BG_SHAP): {9: 100},
        ("california_housing",  _BG_IV):   {9: 100},
    },
    main._SHAP: {
        ("california_housing",  _BG_SHAP): {25: 100},
    },
}


def _adapt(overrides: list[ApproachDOverride], label: str, task_type: TaskType) -> list[ApproachDOverride]:
    """Swap in the m=100 SHAP approach, relabel prerecorded times and apply _MORE_TREES."""
    out = []
    for ov in overrides:
        model_by_D = {D: _relabel(spec) for D, spec in ov.model_by_D.items()}
        more = _MORE_TREES.get(ov.approach, {}).get((label, task_type), {})
        if more:
            base = next(s for s in model_by_D.values() if isinstance(s, ModelSpec))
            for D, T in more.items():
                if T is MEMORY_CRASH:
                    model_by_D[D] = MEMORY_CRASH
                    continue
                hp = {**base.config.hyperparams, "max_depth": D, "n_estimators": T}
                model_by_D[D] = ModelSpec(replace(base.config, hyperparams=hp), base.wrapper)
        out.append(ApproachDOverride(
            approach=_SHAP if ov.approach is main._SHAP else ov.approach,
            full_T=ov.full_T,
            model_by_D=model_by_D,
        ))
    return out


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
                _adapt(overrides_fn(task_type, specs), label, task_type),
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
