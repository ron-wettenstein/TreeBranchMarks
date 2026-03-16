"""
WoodelfHD Depth Sweep Experiment.

Compares three SHAP methods across four datasets and four task types,
sweeping over tree depth D using the ControlledMission API.

Methods
-------
- OriginalWoodelf (OriginalWoodelfApproach) — cube-based (woodelf.simple_woodelf); runs at D=6–15 depending on dataset/task
- WoodelfHD       (WoodelfHDApproach)      — high-depth woodelf; direct woodelf_for_high_depth call, no depth limits
- SHAP         (SHAPApproach)         — shap package reference

Datasets / D sweeps
-------------------
- Fraud Detection  [6, 9, 12, 15, 18, 21]
- HIGGS            [6, 9, 12, 15, 18, 21]
- KDD Cup          [6, 9, 12, 15, 18, 21]
- California Housing [6, 9, 12, 15, 18, 21, 25]

OriginalWoodelf
---------------
  Cube-based (woodelf.simple_woodelf); runs at D=6,9 for all datasets.
  Also runs at D=12 (T=10) and sometimes D=15 (T=1) depending on dataset/task,
  but crashes (MEMORY_CRASH) at D≥18 for SHAP tasks and D≥15 for IV tasks.

WoodelfHD
---------
  Direct woodelf_for_high_depth calls; no internal depth limits or extrapolation.
  MEMORY_CRASH sentinels in this file are the sole crash authority for WoodelfHD.

T values per method/depth/dataset
----------------------------------
Chosen to match the notebooks (minimum T that keeps wall-clock time reasonable).
ControlledMission scales elapsed_s by full_T / actual_T when actual_T < full_T=100.

BG IV / SHAP: shap package does not support background SHAP interactions.
  SHAP receives real models (same as WoodelfHD for that task); approach returns not_supported.

Run from project root:
    python -m benchmarks.woodelfhd_depth_sweep_experiment
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from treebranchmarks.core.cli import run_experiment_cli
from treebranchmarks import Experiment
from treebranchmarks.core.mission import (
    ControlledMission,
    ApproachDOverride,
    ModelSpec,
    MEMORY_CRASH,
    PrerecordedTime,
)
from treebranchmarks.core.model import ModelConfig
from treebranchmarks.core.params import EnsembleType
from treebranchmarks.core.task import TaskType
from treebranchmarks.datasets import (
    FraudDetectionDataset,
    HIGGSDataset,
    IntrusionDetectionDataset,
    CaliforniaHousingDataset,
)
from treebranchmarks.methods.shap_method import SHAPApproach
from treebranchmarks.methods.woodelf_original_and_hd_method import (
    WoodelfHDApproach,
    OriginalWoodelfApproach,
)
from treebranchmarks.models.lightgbm_model import LightGBMWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_ROOT  = Path("cache")
RESULTS_DIR = Path("results")
_FULL_T     = 100

_D_STANDARD = [6, 9, 12, 15, 18, 21]
_D_HOUSING  = [6, 9, 12, 15, 18, 21, 25]

# ---------------------------------------------------------------------------
# Shared stateless approach instances
# ---------------------------------------------------------------------------

_WOODELF_AAAI = OriginalWoodelfApproach()
_WOODELF_HD   = WoodelfHDApproach()
_SHAP         = SHAPApproach(bg_shap_limit=10)

# ---------------------------------------------------------------------------
# LightGBM hyperparameter templates (from notebooks)
# ---------------------------------------------------------------------------

_LGBM_CLS = dict(
    num_leaves=2024, colsample_bytree=0.8, subsample=0.8,
    subsample_freq=1, learning_rate=0.1, min_child_samples=500,
)
_LGBM_REG = dict(
    num_leaves=2024, colsample_bytree=0.8, subsample=0.8,
    subsample_freq=1, learning_rate=0.1, min_child_samples=5,
)


def _lgbm_config(depth: int, n_trees: int, objective: str) -> ModelConfig:
    base = _LGBM_CLS.copy() if objective == "classification" else _LGBM_REG.copy()
    hp = {"max_depth": depth, "n_estimators": n_trees}
    hp.update({k: v for k, v in base.items() if k not in hp})
    return ModelConfig(ensemble_type=EnsembleType.LIGHTGBM, hyperparams=hp, random_state=42)


# ---------------------------------------------------------------------------
# Model spec pools
# ---------------------------------------------------------------------------

@dataclass
class _Specs:
    """Pools of (D → ModelSpec) for T=100, T=10, T=1."""
    s100: dict  # int → ModelSpec
    s10:  dict
    s1:   dict


def _build_pool(D_list: list[int], T: int, objective: str, wrapper: LightGBMWrapper) -> dict:
    return {D: ModelSpec(_lgbm_config(D, T, objective), wrapper) for D in D_list}


def _build_specs(
    D_100: list[int], D_10: list[int], D_1: list[int],
    objective: str, wrapper: LightGBMWrapper,
) -> _Specs:
    return _Specs(
        s100=_build_pool(D_100, 100, objective, wrapper),
        s10 =_build_pool(D_10,  10,  objective, wrapper),
        s1  =_build_pool(D_1,   1,   objective, wrapper),
    )


def _ov(approach, model_by_D: dict) -> ApproachDOverride:
    """Shorthand for ApproachDOverride with full_T=100."""
    return ApproachDOverride(approach=approach, full_T=_FULL_T, model_by_D=model_by_D)


# ---------------------------------------------------------------------------
# Fraud Detection overrides
# Source: background_shap_fraud_data.ipynb, path_dependnet_shap_fraud_data.ipynb
#
# WoodelfHD BG/PD SHAP — D=6-18: T=100; D=21: T=10
# WoodelfHD BG/PD IV   — D=6-15: T=100; D=18: T=10; D=21: CRASH
# WoodelfAAAI BG/PD SHAP — D=6,9: T=100; D=12: T=10; D=15: T=1; D=18,21: CRASH
# WoodelfAAAI BG/PD IV   — D=6,9: T=100; D=12: T=10; D=15,18,21: CRASH
# SHAP BG    — D=6-18: T=100; D=21: T=10
# SHAP PD    — D=6-21: T=100
# SHAP PD IV — D=6,9: T=100; D=12: T=10; D=15-21: T=1
# ---------------------------------------------------------------------------

def _fraud_specs(cache_root: Path) -> _Specs:
    w = LightGBMWrapper(task_type="classification")
    return _build_specs(
        D_100=[6, 9, 12, 15, 18, 21],
        D_10 =[12, 15, 18, 21],
        D_1  =[15, 18, 21],
        objective="classification",
        wrapper=w,
    )


def _fraud_overrides(task_type: TaskType, sp: _Specs) -> list[ApproachDOverride]:
    s100, s10, s1 = sp.s100, sp.s10, sp.s1

    if task_type == TaskType.BACKGROUND_SHAP:
        hd   = {6: s100[6], 9: s100[9], 12: s100[12], 15: s100[15], 18: s100[18], 21: s10[21]}
        aaai = {6: s100[6], 9: s100[9], 12: s10[12],  15: s1[15],   18: MEMORY_CRASH, 21: MEMORY_CRASH}
        shap = {6:s100[6], 9:s100[9], 12:s100[12], 15:s100[15], 18:s100[18], 21:s10[21]}

    elif task_type == TaskType.PATH_DEPENDENT_SHAP:
        hd   = {6: s100[6], 9: s100[9], 12: s100[12], 15: s100[15], 18: s100[18], 21: s10[21]}
        aaai = {6: s100[6], 9: s100[9], 12: s10[12],  15: s1[15],   18: MEMORY_CRASH, 21: MEMORY_CRASH}
        shap = {6: s100[6], 9: s100[9], 12: s100[12], 15: s100[15], 18: s100[18], 21: s100[21]}

    elif task_type == TaskType.BACKGROUND_SHAP_INTERACTIONS:
        hd   = {6: s100[6], 9: s100[9], 12: s100[12], 15: s100[15], 18: s10[18], 21: MEMORY_CRASH}
        aaai = {6: s100[6], 9: s100[9], 12: s10[12],  15: MEMORY_CRASH, 18: MEMORY_CRASH, 21: MEMORY_CRASH}
        shap = {6: s100[6], 9: s100[9], 12: s100[12], 15: s100[15], 18: s100[18], 21: s100[21]}   # not_supported

    else:  # PATH_DEPENDENT_INTERACTIONS
        hd   = {6: s100[6], 9: s100[9], 12: s100[12], 15: s10[15], 18: s10[18], 21: MEMORY_CRASH}
        aaai = {6: s100[6], 9: s100[9], 12: s10[12],  15: MEMORY_CRASH, 18: MEMORY_CRASH, 21: MEMORY_CRASH}
        shap = {6: s100[6], 9: s100[9], 12: s10[12],  15: s1[15],  18: s1[18], 21: s1[21]}

    return [_ov(_WOODELF_HD, hd), _ov(_WOODELF_AAAI, aaai), _ov(_SHAP, shap)]


# ---------------------------------------------------------------------------
# HIGGS overrides
# Source: background_and_path_dependent_HIGGS_data.ipynb
#
# WoodelfHD BG/PD SHAP — D=6:T100; D=9:T10; D=12-21:T1 (huge dataset, memory pressure)
# WoodelfHD BG IV       — D=6:T100; D=9:T10; D=12,15:T1; D=18,21:CRASH
# WoodelfHD PD IV       — D=6:T100; D=9-15:T10; D=18,21:CRASH
# WoodelfAAAI           — D=6:T100; D=9:T10 (D≥12 crash)
# SHAP BG SHAP          — D=6-21:T1 (200k rows, slow with T=100)
# SHAP PD SHAP          — D=6:T10; D=9-21:T1
# SHAP PD IV            — D=6:T100; D=9:T10; D=12-21:T1
# ---------------------------------------------------------------------------

def _higgs_specs(cache_root: Path) -> _Specs:
    w = LightGBMWrapper(task_type="classification")
    return _build_specs(
        D_100=[6],
        D_10 =[6, 9, 12, 15],
        D_1  =[6, 9, 12, 15, 18, 21],
        objective="classification",
        wrapper=w,
    )


def _higgs_overrides(task_type: TaskType, sp: _Specs) -> list[ApproachDOverride]:
    s100, s10, s1 = sp.s100, sp.s10, sp.s1
    aaai = {6: s100[6], 9: s10[9], 12: MEMORY_CRASH, 15: MEMORY_CRASH, 18: MEMORY_CRASH, 21: MEMORY_CRASH}

    if task_type == TaskType.BACKGROUND_SHAP:
        hd   = {6: s100[6], 9: s10[9], 12: s1[12], 15: s1[15], 18: s1[18], 21: s1[21]}
        shap = {6: s1[6],   9: s1[9],  12: s1[12], 15: s1[15], 18: s1[18], 21: s1[21]}

    elif task_type == TaskType.PATH_DEPENDENT_SHAP:
        hd   = {6: s100[6], 9: s10[9], 12: s1[12], 15: s1[15], 18: s1[18], 21: s1[21]}
        shap = {6: s10[6],  9: s1[9],  12: s1[12], 15: s1[15], 18: s1[18], 21: s1[21]}

    elif task_type == TaskType.BACKGROUND_SHAP_INTERACTIONS:
        # WoodelfHD IV: D=18: T=1; D=21: MEMORY_CRASH
        hd   = {6: s100[6], 9: s10[9], 12: s1[12], 15: s1[15], 18: s1[18],  21: MEMORY_CRASH}
        shap = {6: s100[6], 9: s10[9], 12: s1[12], 15: s1[15], 18: s1[18],  21: s1[21]}   # not_supported

    else:  # PATH_DEPENDENT_INTERACTIONS
        # WoodelfHD PD IV: D=9,12,15: T=10; D=18: T=1; D=21: MEMORY_CRASH
        hd   = {6: s100[6], 9: s10[9], 12: s10[12], 15: s1[15], 18: s1[18], 21: MEMORY_CRASH}
        shap = {6: s100[6], 9: s10[9], 12: s1[12],  15: s1[15],  18: s1[18], 21: s1[21]}

    return [_ov(_WOODELF_HD, hd), _ov(_WOODELF_AAAI, aaai), _ov(_SHAP, shap)]


# ---------------------------------------------------------------------------
# KDD Cup (Intrusion Detection) overrides
# Source: background_and_path_dependent_KDD_data.ipynb
#
# WoodelfHD BG SHAP  — D=6,9:T100; D=12-18:T10; D=21:T1
# WoodelfHD PD SHAP  — D=6:T100; D=9-18:T10; D=21:T1
# WoodelfHD BG/PD IV — D=6,9:T100; D=12,15:T10; D=18,21:CRASH
# WoodelfAAAI BG/PD SHAP — D=6:T100; D=9:T10 (D≥12 crash)
# WoodelfAAAI BG/PD IV   — D=6,9:T100 (D≥12 crash)
# SHAP BG SHAP       — D=6-21:T10
# SHAP PD SHAP       — D=6-15:T10; D=18,21:T1
# SHAP PD IV         — D=6-21:T10
# ---------------------------------------------------------------------------

def _kdd_specs(cache_root: Path) -> _Specs:
    w = LightGBMWrapper(task_type="classification")
    return _build_specs(
        D_100=[6, 9],
        D_10 =[6, 9, 12, 15, 18, 21],
        D_1  =[12, 15, 18, 21],
        objective="classification",
        wrapper=w,
    )


def _kdd_overrides(task_type: TaskType, sp: _Specs) -> list[ApproachDOverride]:
    s100, s10, s1 = sp.s100, sp.s10, sp.s1

    if task_type == TaskType.BACKGROUND_SHAP:
        hd   = {6: s100[6], 9: s100[9], 12: s10[12], 15: s10[15], 18: s10[18], 21: s1[21]}
        aaai = {6: s100[6], 9: s10[9],  12: s1[12],  15: s1[15],  18: MEMORY_CRASH, 21: MEMORY_CRASH}
        shap = {6: s10[6],  9: s10[9],  12: s10[12], 15: s10[15], 18: s10[18], 21: s10[21]}

    elif task_type == TaskType.PATH_DEPENDENT_SHAP:
        hd   = {6: s100[6], 9: s10[9],  12: s10[12], 15: s10[15], 18: s10[18], 21: s1[21]}
        aaai = {6: s100[6], 9: s10[9],  12: s1[12],  15: s1[15],  18: MEMORY_CRASH, 21: MEMORY_CRASH}
        shap = {6: s10[6],  9: s10[9],  12: s10[12], 15: s10[15], 18: s1[18],  21: s1[21]}

    elif task_type == TaskType.BACKGROUND_SHAP_INTERACTIONS:
        # WoodelfHD IV: D=18: T=10; D=21: MEMORY_CRASH
        hd   = {6: s100[6], 9: s100[9], 12: s10[12], 15: s10[15], 18: s10[18], 21: MEMORY_CRASH}
        aaai = {6: s100[6], 9: s100[9], 12: s1[12],  15: MEMORY_CRASH, 18: MEMORY_CRASH, 21: MEMORY_CRASH}
        shap = {6: s100[6], 9: s100[9], 12: s10[12], 15: s10[15], 18: s10[18], 21: s10[21]}  # not_supported

    else:  # PATH_DEPENDENT_INTERACTIONS
        # WoodelfHD IV: D=18: T=10; D=21: MEMORY_CRASH
        hd   = {6: s100[6], 9: s100[9], 12: s10[12], 15: s10[15], 18: s1[18], 21: MEMORY_CRASH}
        aaai = {6: s100[6], 9: s100[9], 12: s1[12],  15: MEMORY_CRASH, 18: MEMORY_CRASH, 21: MEMORY_CRASH}
        shap = {6: s10[6],  9: s10[9],  12: s10[12], 15: s10[15], 18: s10[18], 21: s10[21]}

    return [_ov(_WOODELF_HD, hd), _ov(_WOODELF_AAAI, aaai), _ov(_SHAP, shap)]


# ---------------------------------------------------------------------------
# California Housing overrides
# Source: background_and_path_dependent_housing_data.ipynb
#
# WoodelfHD      — all tasks: T=100 for D=6-25 (small dataset, no crash)
# WoodelfAAAI BG SHAP    — D=6:T100; D=9:T10; D=12,15:T1; D=18+: CRASH
# WoodelfAAAI BG/PD IV   — D=6,9:T100; D=12:T1; D=15+: CRASH
# WoodelfAAAI PD SHAP    — D=6,9:T100; D=12,15:T1; D=18+: CRASH
# WoodelfAAAI PD IV      — D=6,9:T100; D=12:T10; D=15+: CRASH
# SHAP BG/PD SHAP        — T=100, D=6-21
# SHAP BG IV             — T=100, D=6-21 (not_supported)
# SHAP PD IV             — D=6,9:T100; D=12:T10; D=15,18,21:T1
# ---------------------------------------------------------------------------

def _housing_specs(cache_root: Path) -> _Specs:
    w = LightGBMWrapper(task_type="regression")
    return _build_specs(
        D_100=[6, 9, 12, 15, 18, 21, 25],
        D_10 =[9, 12, 25],
        D_1  =[15, 18, 21, 25],
        objective="regression",
        wrapper=w,
    )


_AAAI_CRASH_18_25 = {18: MEMORY_CRASH, 21: MEMORY_CRASH, 25: MEMORY_CRASH}


def _housing_overrides(task_type: TaskType, sp: _Specs, D_values: list[int]) -> list[ApproachDOverride]:
    s100, s10, s1 = sp.s100, sp.s10, sp.s1
    # hd: T=100 for all depths (housing is small, no crashes)
    hd = {D: s100[D] for D in D_values}
    shap_models = {6: s100[6], 9: s100[9], 12: s100[12], 15: s100[15], 18: s100[18], 21: s100[21], 25: s10[25]}

    _desc = "run in a jupyter notebook beforehand, for depth>9 used T=10 or T=1 was extrapolated to T=100"
    def _pre(t): return PrerecordedTime(t, estimation_description=_desc)

    if task_type == TaskType.BACKGROUND_SHAP:
        aaai = {6: s100[6], 9: s10[9], 12: _pre(152460),  15: s1[15], **_AAAI_CRASH_18_25}
        shap = shap_models

    elif task_type == TaskType.PATH_DEPENDENT_SHAP:
        aaai = {6: s100[6], 9: _pre(864),   12: _pre(106798),  15: _pre(1532740), **_AAAI_CRASH_18_25}
        shap = shap_models

    elif task_type == TaskType.BACKGROUND_SHAP_INTERACTIONS:
        aaai = {6: s100[6], 9: _pre(993),   12: _pre(112663),  15: MEMORY_CRASH, **_AAAI_CRASH_18_25}
        shap = shap_models  # not_supported

    else:  # PATH_DEPENDENT_INTERACTIONS
        aaai = {6: s100[6], 9: _pre(1143),  12: _pre(123291),  15: MEMORY_CRASH, **_AAAI_CRASH_18_25}
        shap = {6: s100[6], 9: s100[9], 12: s10[12], 15: s1[15], 18: s1[18], 21: s1[21], 25: s1[25]}

    return [_ov(_WOODELF_HD, hd), _ov(_WOODELF_AAAI, aaai), _ov(_SHAP, shap)]


# ---------------------------------------------------------------------------
# Mission builder
# ---------------------------------------------------------------------------

def _mission(
    name: str,
    dataset,
    D_values: list[int],
    overrides: list[ApproachDOverride],
    task_type: TaskType,
    n: int,
    m: int,
    cache_root: Path,
) -> ControlledMission:
    return ControlledMission(
        name=name,
        dataset=dataset,
        D_values=D_values,
        approach_overrides=overrides,
        task_types=[task_type],
        n=n,
        m=m,
        cache_root=cache_root,
    )


def _build_dataset_missions(
    label: str,
    dataset,
    D_shap: list[int],
    D_iv: list[int],
    specs: _Specs,
    overrides_fn,   # callable(task_type, specs, *extra) -> list[ApproachDOverride]
    n_pd: int, n_bg: int, n_iv: int, m_bg: int,
    cache_root: Path,
    overrides_kwargs: dict | None = None,
) -> list[ControlledMission]:
    """Create the four standard missions (PD SHAP, BG SHAP, PD IV, BG IV)."""
    kw = overrides_kwargs or {}

    return [
        _mission(
            f"{label} PD SHAP sweep_D", dataset, D_shap,
            overrides_fn(TaskType.PATH_DEPENDENT_SHAP, specs, **kw),
            TaskType.PATH_DEPENDENT_SHAP, n_pd, 0, cache_root,
        ),
        _mission(
            f"{label} BG SHAP sweep_D", dataset, D_shap,
            overrides_fn(TaskType.BACKGROUND_SHAP, specs, **kw),
            TaskType.BACKGROUND_SHAP, n_bg, m_bg, cache_root,
        ),
        _mission(
            f"{label} PD IV sweep_D", dataset, D_iv,
            overrides_fn(TaskType.PATH_DEPENDENT_INTERACTIONS, specs, **kw),
            TaskType.PATH_DEPENDENT_INTERACTIONS, n_iv, 0, cache_root,
        ),
        _mission(
            f"{label} BG IV sweep_D", dataset, D_iv,
            overrides_fn(TaskType.BACKGROUND_SHAP_INTERACTIONS, specs, **kw),
            TaskType.BACKGROUND_SHAP_INTERACTIONS, n_iv, m_bg, cache_root,
        ),
    ]


# ---------------------------------------------------------------------------
# Top-level builders
# ---------------------------------------------------------------------------

def build_missions(cache_root: Path = CACHE_ROOT) -> list[ControlledMission]:
    missions = []

    # Fraud Detection
    missions += _build_dataset_missions(
        label="fraud_detection",
        dataset=FraudDetectionDataset(),
        D_shap=_D_STANDARD, D_iv=_D_STANDARD,
        specs=_fraud_specs(cache_root),
        overrides_fn=_fraud_overrides,
        n_pd=118_108, n_bg=118_108, n_iv=10_000, m_bg=472_432,
        cache_root=cache_root,
    )

    # HIGGS
    missions += _build_dataset_missions(
        label="higgs",
        dataset=HIGGSDataset(),
        D_shap=_D_STANDARD, D_iv=_D_STANDARD,
        specs=_higgs_specs(cache_root),
        overrides_fn=_higgs_overrides,
        n_pd=2_200_000, n_bg=2_200_000, n_iv=10_000, m_bg=8_800_000,
        cache_root=cache_root,
    )

    # KDD Cup
    missions += _build_dataset_missions(
        label="intrusion_detection",
        dataset=IntrusionDetectionDataset(),
        D_shap=_D_STANDARD, D_iv=_D_STANDARD,
        specs=_kdd_specs(cache_root),
        overrides_fn=_kdd_overrides,
        n_pd=2_984_154, n_bg=2_984_154, n_iv=10_000, m_bg=4_898_431,
        cache_root=cache_root,
    )

    # California Housing — housing overrides_fn takes extra D_values kwarg
    housing_specs = _housing_specs(cache_root)
    housing_ds    = CaliforniaHousingDataset()
    for task_type, label_sfx, n, m in [
        (TaskType.PATH_DEPENDENT_SHAP,         "PD SHAP", 4128, 0),
        (TaskType.BACKGROUND_SHAP,             "BG SHAP", 4128, 16512),
        (TaskType.PATH_DEPENDENT_INTERACTIONS, "PD IV",   4128, 0),
        (TaskType.BACKGROUND_SHAP_INTERACTIONS,"BG IV",   4128, 16512),
    ]:
        missions.append(_mission(
            f"california_housing {label_sfx} sweep_D",
            housing_ds, _D_HOUSING,
            _housing_overrides(task_type, housing_specs, _D_HOUSING),
            task_type, n, m, cache_root,
        ))

    return missions


def build_experiment() -> Experiment:
    return Experiment(
        name="woodelfhd_depth_sweep_experiment",
        missions=build_missions(),
        results_dir=RESULTS_DIR,
        force_rerun=False,
        delete_dataset_cache=False,
        delete_model_cache=False,
        delete_results=False,
    )


if __name__ == "__main__":
    run_experiment_cli(build_experiment)
