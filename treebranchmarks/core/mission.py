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
from treebranchmarks.core.dataset import Dataset
from treebranchmarks.core.model import ModelConfig, ModelWrapper, TrainedModel
from treebranchmarks.core.task import Task, TaskResult


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

        X, y = cfg.dataset.load()

        meta = {
            "dataset": {
                "name": cfg.dataset.name,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "columns": list(X.columns),
            },
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

        for model_config, wrapper in cfg.model_wrappers.items():
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
                    X_background = (
                        X.iloc[idx[n : n + m]].reset_index(drop=True)
                        if m > 0
                        else None
                    )

                    print(f"\n  > model={model_config}  n={n}  m={m}")

                    for task in cfg.tasks:
                        task_result = task.run(
                            trained, X_explain, X_background,
                            method_cache=method_cache,
                            mission_name=self.name,
                        )
                        result.task_results.append(task_result)

        return result
