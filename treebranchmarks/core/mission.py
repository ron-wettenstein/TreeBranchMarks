"""
Mission: a single parameter sweep combining one dataset, one or more model
configs, one or more tasks, and lists of n_values / m_values to sweep over.

Design: each mission should have exactly ONE free variable (n, m, or D/depth).
The mission name is auto-generated to reflect the free variable if not provided.

The outer loop is:
    model_config × n_value (sorted asc) × m_value × task × approach

Caching
-------
Dataset loading and model training are cached via the wrappers' own mechanisms.
Mission results are NOT cached here — that is the Experiment's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from treebranchmarks.cache.method_cache import MethodResultCache
from treebranchmarks.core.approach import Approach
from treebranchmarks.core.dataset import Dataset
from treebranchmarks.core.model import ModelConfig, ModelWrapper, TrainedModel
from treebranchmarks.core.params import TreeParameters
from treebranchmarks.core.task import ApproachResult, Task, TaskResult, TaskType


# ---------------------------------------------------------------------------
# MissionConfig
# ---------------------------------------------------------------------------

@dataclass
class MissionConfig:
    """
    Declarative description of a parameter sweep.

    Design intent: exactly one of {n_values, m_values, model_wrappers (via
    varying depth)} should have more than one distinct value.  The mission name
    is auto-generated from the free variable when ``name`` is left empty.

    Parameters
    ----------
    dataset : Dataset
    model_wrappers : dict[ModelConfig, ModelWrapper]
    tasks : list[Task]
    n_values : list[int]
    m_values : list[int]
        For tasks that don't use a background, pass [0].
    name : str
        Human-readable label shown in the HTML report.  Auto-generated if "".
    random_state : int
    cache_root : Path
    """

    dataset: Dataset
    model_wrappers: dict[ModelConfig, ModelWrapper]
    tasks: list[Task]
    n_values: list[int]
    m_values: list[int]
    name: str = ""
    random_state: int = 42
    cache_root: Path = Path("cache")


# ---------------------------------------------------------------------------
# Name auto-generation
# ---------------------------------------------------------------------------

def _auto_name(config: MissionConfig) -> str:
    """Infer a mission name from the dimension(s) that vary."""
    parts = []

    if len(set(config.n_values)) > 1:
        parts.append(f"sweep_n {sorted(set(config.n_values))}")

    meaningful_m = [m for m in config.m_values if m > 0]
    if len(set(meaningful_m)) > 1:
        parts.append(f"sweep_m {sorted(set(config.m_values))}")

    depths = [mc.hyperparams.get("max_depth") for mc in config.model_wrappers
              if mc.hyperparams.get("max_depth") is not None]
    if len(set(depths)) > 1:
        parts.append(f"sweep_D {sorted(set(depths))}")

    ensembles = {mc.ensemble_type.value for mc in config.model_wrappers}
    if len(ensembles) > 1:
        parts.append(f"sweep_ensemble {sorted(ensembles)}")

    if not parts:
        # Single point — describe the fixed values
        n = config.n_values[0] if config.n_values else "?"
        m = config.m_values[0] if config.m_values else "?"
        parts.append(f"n={n} m={m}")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# MissionResult
# ---------------------------------------------------------------------------

@dataclass
class MissionResult:
    config: MissionConfig
    mission_name: str
    meta: dict = field(default_factory=dict)
    task_results: list[TaskResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        dataset_name = (
            self.config.dataset.name
            if self.config is not None
            else getattr(self, "_dataset_name", "unknown")
        )
        return {
            "dataset": dataset_name,
            "mission_name": self.mission_name,
            "meta": self.meta,
            "task_results": [r.as_dict() for r in self.task_results],
        }


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------

class Mission:
    """
    Executes a parameter sweep and returns all TaskResults.

    Usage
    -----
    >>> mission = Mission(config)
    >>> result = mission.run()
    """

    def __init__(self, config: MissionConfig) -> None:
        self.config = config
        self.name: str = config.name or _auto_name(config)

    def run(
        self,
        method_cache: Optional[MethodResultCache] = None,
    ) -> MissionResult:
        """
        Execute the parameter sweep.

        Parameters
        ----------
        method_cache : MethodResultCache | None
            If provided, each approach result is looked up in the cache before
            running and written to the cache after running.  Methods whose cache
            was cleared (via ``force_rerun_methods``) will simply miss and be
            re-measured.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)

        n_values_sorted = sorted(cfg.n_values)

        print(f"\n{'='*60}")
        print(f"Mission: {self.name}")
        print(f"  dataset       : {cfg.dataset.name}")
        print(f"  model configs : {len(cfg.model_wrappers)}")
        print(f"  n_values      : {n_values_sorted}")
        print(f"  m_values      : {cfg.m_values}")
        print(f"  tasks         : {[t.name for t in cfg.tasks]}")
        print(f"{'='*60}")

        # Determine which models actually need to run (not fully cached).
        # For models whose runs are all cached we only need the lightweight
        # meta.json sidecar; the heavy model artifact and the dataset itself
        # are never loaded for those models.
        models_needing_run: list[tuple] = []      # (model_config, wrapper)
        fully_cached_models: list[tuple] = []     # (model_config, base_params)

        for model_config, wrapper in cfg.model_wrappers.items():
            if method_cache is not None:
                base_params = wrapper.load_params_only(cfg.cache_root, cfg.dataset.name, model_config)
                if base_params is not None and all(
                    method_cache.all_approaches_cached(
                        task.approaches, self.name, task.name,
                        base_params.with_run_params(n=n, m=m),
                    )
                    for n in n_values_sorted
                    for m in cfg.m_values
                    for task in cfg.tasks
                ):
                    fully_cached_models.append((model_config, base_params))
                    continue
            models_needing_run.append((model_config, wrapper))

        # Only load the dataset when at least one model still needs to run.
        X = y = None
        if models_needing_run:
            X, y = cfg.dataset.load()

        # Build the meta dict — dataset section requires a loaded X.
        # For fully-cached missions use the first cached model's params for
        # feature count / column list (read from meta.json, not from X).
        if X is not None:
            dataset_meta = {
                "name": cfg.dataset.name,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "columns": list(X.columns),
            }
        elif fully_cached_models:
            bp = fully_cached_models[0][1]
            details = cfg.dataset.dump_details()
            dataset_meta = {
                "name": cfg.dataset.name,
                "n_samples": details.get("n_samples"),
                "n_features": bp.F,
                "columns": details.get("columns", []),
            }
        else:
            dataset_meta = {"name": cfg.dataset.name}

        meta = {
            "dataset": dataset_meta,
            "models": [
                {
                    "ensemble_type": mc.ensemble_type.value,
                    "hyperparams": mc.hyperparams,
                    "random_state": mc.random_state,
                }
                for mc in cfg.model_wrappers
            ],
            "tasks": [
                {
                    "name": t.name,
                    "approaches": [
                        {"name": a.name, "description": a.description}
                        for a in t.approaches
                    ],
                }
                for t in cfg.tasks
            ],
            "n_values": sorted(cfg.n_values),
            "m_values": cfg.m_values,
        }

        result = MissionResult(config=cfg, mission_name=self.name, meta=meta)

        # Reconstruct results for fully-cached models directly from cache.
        for model_config, base_params in fully_cached_models:
            print(f"\n  [model:{model_config.ensemble_type.value}] All runs cached — skipping model load.")
            for n in n_values_sorted:
                for m in cfg.m_values:
                    params = base_params.with_run_params(n=n, m=m)
                    for task in cfg.tasks:
                        approach_results = {
                            approach.name: method_cache.get(
                                approach, self.name, task.name, params
                            )
                            for approach in task.approaches
                        }
                        result.task_results.append(
                            TaskResult(
                                task_name=task.name,
                                params=params,
                                approach_results=approach_results,
                            )
                        )

        # Run models that have at least one uncached result.
        for model_config, wrapper in models_needing_run:
            trained = wrapper.load_or_train(
                dataset_name=cfg.dataset.name,
                X=X, y=y,
                config=model_config,
                cache_root=cfg.cache_root,
            )

            for n in n_values_sorted:
                for m in cfg.m_values:
                    idx = rng.permutation(len(X))
                    X_explain = X.iloc[idx[:n]].reset_index(drop=True)
                    if m == 0:
                        X_background = None
                    elif n + m <= len(X):
                        X_background = X.iloc[idx[n : n + m]].reset_index(drop=True)
                    else:
                        # n + m exceeds dataset size — allow overlap with a fresh permutation
                        bg_idx = rng.permutation(len(X))[:m]
                        X_background = X.iloc[bg_idx].reset_index(drop=True)

                    print(f"\n  > model={model_config}  n={n}  m={m}")

                    for task in cfg.tasks:
                        task_result = task.run(
                            trained, X_explain, X_background,
                            method_cache=method_cache,
                            mission_name=self.name,
                        )
                        result.task_results.append(task_result)

        return result


# ---------------------------------------------------------------------------
# ControlledMission — per-approach, per-D model configuration
# ---------------------------------------------------------------------------

#: Sentinel value for ApproachDOverride.model_by_D — forces memory_crash.
MEMORY_CRASH = "memory_crash"


@dataclass
class PrerecordedTime:
    """
    A prerecorded elapsed time to use in place of actually running an approach.

    Use this in ApproachDOverride.model_by_D when you already know how long
    the approach took (e.g. a previous 2-hour run) and don't want to re-run it.

    Parameters
    ----------
    elapsed_s : float
        The known wall-clock time in seconds.
    estimation_description : str
        Optional note explaining the source of the prerecorded value.
    """
    elapsed_s: float
    estimation_description: str = ""


@dataclass
class ModelSpec:
    """Pairs a ModelConfig with its ModelWrapper for use in ControlledMission."""
    config: ModelConfig
    wrapper: ModelWrapper


@dataclass
class ApproachDOverride:
    """
    Specifies which model each approach uses for each D value.

    Parameters
    ----------
    approach : Approach
    full_T : int
        The "reference" tree count.  When the model at a given D has fewer
        than full_T trees, elapsed time is scaled by full_T / actual_T and
        is_estimated is set to True.
    model_by_D : dict[int, ModelSpec | str]
        Maps D value → ModelSpec to use, or MEMORY_CRASH to force a crash result.
        D values absent from the map produce MEMORY_CRASH results.
    """
    approach: Approach
    full_T: int
    model_by_D: dict


class ControlledMission:
    """
    A D-sweep mission where each approach can use a different model per D value.

    Useful when some approaches are too slow at high depth: provide a
    reduced-tree ModelSpec and the mission scales elapsed_s by full_T / actual_T
    automatically (is_estimated=True).

    Example
    -------
    ::

        from treebranchmarks.core.mission import (
            ControlledMission, ApproachDOverride, ModelSpec, MEMORY_CRASH
        )
        from treebranchmarks.core.task import TaskType

        mission = ControlledMission(
            name="D sweep controlled",
            dataset=FraudDetectionDataset(),
            D_values=[6, 9, 12],
            approach_overrides=[
                ApproachDOverride(
                    approach=WoodelfApproach(),
                    full_T=100,
                    model_by_D={
                        6: ModelSpec(lgbm_cfg_d6_100t, LightGBMModel()),
                        9: ModelSpec(lgbm_cfg_d9_10t,  LightGBMModel()),   # extrap ×10
                       12: ModelSpec(lgbm_cfg_d12_10t, LightGBMModel()),
                    },
                ),
                ApproachDOverride(
                    approach=WoodelfAAAIApproach(),
                    full_T=100,
                    model_by_D={
                        6: ModelSpec(lgbm_cfg_d6_100t, LightGBMModel()),
                        9: ModelSpec(lgbm_cfg_d9_1t,   LightGBMModel()),   # extrap ×100
                       12: MEMORY_CRASH,
                    },
                ),
            ],
            task_types=[TaskType.BACKGROUND_SHAP],
            n=1000,
            m=100,
        )
    """

    def __init__(
        self,
        name: str,
        dataset: Dataset,
        D_values: list,
        approach_overrides: list,
        task_types: list,
        n: int,
        m: int,
        random_state: int = 42,
        cache_root: Path = Path("cache"),
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.D_values = sorted(D_values)
        self.approach_overrides: list[ApproachDOverride] = approach_overrides
        self.task_types: list[TaskType] = task_types
        self.n = n
        self.m = m
        self.random_state = random_state
        self.cache_root = cache_root

    def run(
        self,
        method_cache: Optional[MethodResultCache] = None,
    ) -> MissionResult:
        rng = np.random.default_rng(self.random_state)

        print(f"\n{'='*60}")
        print(f"Mission: {self.name}")
        print(f"  dataset   : {self.dataset.name}")
        print(f"  D_values  : {self.D_values}")
        print(f"  n={self.n}  m={self.m}")
        print(f"  tasks     : {[t.display_name for t in self.task_types]}")
        print(f"{'='*60}")

        X, y = self.dataset.load()

        meta = {
            "dataset": {
                "name": self.dataset.name,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "columns": list(X.columns),
            },
            "n_values": [self.n],
            "m_values": [self.m],
        }

        result = MissionResult(config=None, mission_name=self.name, meta=meta)
        result._dataset_name = self.dataset.name

        # Cache trained models by ModelSpec identity to avoid redundant training.
        _trained_cache: dict[int, TrainedModel] = {}

        def _get_trained(spec: ModelSpec) -> TrainedModel:
            key = id(spec)
            if key not in _trained_cache:
                _trained_cache[key] = spec.wrapper.load_or_train(
                    dataset_name=self.dataset.name,
                    X=X, y=y,
                    config=spec.config,
                    cache_root=self.cache_root,
                )
            return _trained_cache[key]

        def _sample(D: int):
            idx = rng.permutation(len(X))
            X_exp = X.iloc[idx[:self.n]].reset_index(drop=True)
            if self.m == 0:
                X_bg = None
            elif self.n + self.m <= len(X):
                X_bg = X.iloc[idx[self.n : self.n + self.m]].reset_index(drop=True)
            else:
                bg_idx = rng.permutation(len(X))[:self.m]
                X_bg = X.iloc[bg_idx].reset_index(drop=True)
            return X_exp, X_bg

        for D in self.D_values:
            X_explain, X_background = _sample(D)
            print(f"\n  > D={D}  n={self.n}  m={self.m}")

            for task_type in self.task_types:
                approach_results: dict[str, ApproachResult] = {}
                reference_params: Optional[TreeParameters] = None

                for override in self.approach_overrides:
                    approach = override.approach
                    model_spec = override.model_by_D.get(D, MEMORY_CRASH)
                    method_name = getattr(getattr(approach, "method", None), "name", "")

                    if model_spec == MEMORY_CRASH:
                        approach_results[approach.name] = ApproachResult(
                            approach_name=approach.name,
                            running_time=0.0,
                            std_time_s=0.0,
                            is_estimated=False,
                            error=None,
                            method=method_name,
                            memory_crash=True,
                        )
                        print(f"  [approach:{approach.name}] MEMORY CRASH (configured)")
                        continue

                    if isinstance(model_spec, PrerecordedTime):
                        ar = ApproachResult(
                            approach_name=approach.name,
                            running_time=model_spec.elapsed_s,
                            std_time_s=0.0,
                            is_estimated=True,
                            error=None,
                            method=method_name,
                            estimation_description=model_spec.estimation_description or "prerecorded time",
                        )
                        approach_results[approach.name] = ar
                        print(f"  [approach:{approach.name}] PRERECORDED={ar.running_time:.3f}s")
                        continue

                    trained = _get_trained(model_spec)
                    actual_T = trained.params.T
                    full_T = override.full_T

                    # Build reference params using full_T so all approaches are
                    # compared on the same footing in the HTML report.
                    base_params = trained.params.with_run_params(
                        n=self.n, m=self.m if self.m > 0 else 0
                    )
                    ref_params = TreeParameters(
                        T=full_T,
                        D=base_params.D,
                        L=base_params.L,
                        F=base_params.F,
                        ensemble_type=base_params.ensemble_type,
                        n=base_params.n,
                        m=base_params.m,
                    )
                    if reference_params is None:
                        reference_params = ref_params

                    # Check method cache.
                    if method_cache is not None:
                        cached = method_cache.get(
                            approach, self.name, task_type.display_name, ref_params
                        )
                        if cached is not None:
                            approach_results[approach.name] = cached
                            print(f"  [approach:{approach.name}] CACHED={cached.running_time:.3f}s")
                            continue

                    # Time the approach via Task's internal helper.
                    _task = Task(
                        task_type=task_type,
                        approaches=[approach],
                        cache_root=self.cache_root,
                    )
                    ar = _task._time_approach(
                        approach, trained, X_explain, X_background, ref_params
                    )

                    # Scale if running on fewer trees than full_T.
                    if (
                        actual_T < full_T
                        and not ar.memory_crash
                        and not ar.not_supported
                        and not ar.runtime_error
                    ):
                        scale = full_T / actual_T
                        extrap_note = f"ran {actual_T} of {full_T} trees, extrapolated ×{scale:.0f}"
                        desc = (
                            f"{ar.estimation_description}; {extrap_note}"
                            if ar.estimation_description
                            else extrap_note
                        )
                        ar = ApproachResult(
                            approach_name=ar.approach_name,
                            running_time=ar.running_time * scale,
                            std_time_s=ar.std_time_s * scale,
                            is_estimated=True,
                            error=ar.error,
                            method=ar.method,
                            estimation_description=desc,
                        )

                    approach_results[approach.name] = ar

                    if method_cache is not None and not ar.error:
                        method_cache.put(
                            approach, self.name, task_type.display_name, ref_params, ar
                        )

                    if ar.runtime_error:
                        print(f"  [approach:{approach.name}] RUNTIME ERROR: {ar.error}")
                    elif ar.is_estimated:
                        print(f"  [approach:{approach.name}] ESTIMATED={ar.running_time:.3f}s")
                    else:
                        print(
                            f"  [approach:{approach.name}] "
                            f"mean={ar.running_time:.3f}s ± {ar.std_time_s:.3f}s"
                        )

                if reference_params is None:
                    # All approaches crashed — no params available; skip.
                    continue

                result.task_results.append(
                    TaskResult(
                        task_name=task_type.display_name,
                        params=reference_params,
                        approach_results=approach_results,
                    )
                )

        return result
