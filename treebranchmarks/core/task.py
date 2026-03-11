"""
Task: times all approaches for a given (model, X_explain, X_background) triple.

A Task owns a list of Approach objects and a TaskType that determines which
task method to call on each approach.  When run(), it:
  1. Runs each approach n_repeats times, recording wall-clock time
  2. Returns a TaskResult

TaskResult / ApproachResult are plain dataclasses that serialize cleanly to JSON.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from treebranchmarks.core.approach import Approach
from treebranchmarks.core.model import TrainedModel
from treebranchmarks.core.params import TreeParameters

if TYPE_CHECKING:
    from treebranchmarks.cache.method_cache import MethodResultCache


# ---------------------------------------------------------------------------
# TaskType
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    PATH_DEPENDENT_SHAP          = "path_dependent_shap"
    PATH_DEPENDENT_INTERACTIONS  = "path_dependent_interactions"
    BACKGROUND_SHAP              = "background_shap"
    BACKGROUND_SHAP_INTERACTIONS = "background_shap_interactions"

    @property
    def display_name(self) -> str:
        return {
            TaskType.PATH_DEPENDENT_SHAP:          "Path-Dependent SHAP",
            TaskType.PATH_DEPENDENT_INTERACTIONS:  "Path-Dependent SHAP Interactions",
            TaskType.BACKGROUND_SHAP:              "Background SHAP",
            TaskType.BACKGROUND_SHAP_INTERACTIONS: "Background SHAP Interactions",
        }[self]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ApproachResult:
    approach_name: str
    running_time: float             # measured mean, or linear extrapolation if is_estimated
    std_time_s: float               # 0.0 when is_estimated=True
    is_estimated: bool              # True = run was skipped; time is linearly extrapolated
    error: Optional[str]            # filled if the approach raised an exception
    method: str = ""                # method.name — e.g. "shap" | "woodelf"
    not_supported: bool = False     # True = approach cannot handle this input; score = 0
    memory_crash: bool = False      # True = approach would exhaust memory; score = 0
    runtime_error: bool = False     # True = approach raised an exception; score = 0
    estimation_description: str = ""  # human-readable note on how estimation was done

    def as_dict(self) -> dict:
        return {
            "approach_name": self.approach_name,
            "running_time": self.running_time,
            "std_time_s": self.std_time_s,
            "is_estimated": self.is_estimated,
            "error": self.error,
            "method": self.method,
            "not_supported": self.not_supported,
            "memory_crash": self.memory_crash,
            "runtime_error": self.runtime_error,
            "estimation_description": self.estimation_description,
        }


@dataclass
class TaskResult:
    task_name: str
    params: TreeParameters
    approach_results: dict[str, ApproachResult]  # keyed by approach.name

    def as_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "params": self.params.as_dict(),
            "approach_results": {
                k: v.as_dict() for k, v in self.approach_results.items()
            },
        }


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class Task:
    """
    Groups one or more Approaches under a task type and times them.

    Parameters
    ----------
    task_type : TaskType
        Which benchmark task to run on each approach.
    approaches : list[Approach]
        The algorithm implementations to benchmark.
    n_repeats : int
        How many timed repetitions per approach (default 3).
    cache_root : Path
        Reserved for future calibration lookup.
    name : str | None
        Display name; defaults to task_type.display_name.
    """

    def __init__(
        self,
        task_type: TaskType,
        approaches: list[Approach],
        n_repeats: int = 3,
        cache_root: Path = Path("cache"),
        name: Optional[str] = None,
    ) -> None:
        self.task_type = task_type
        self.approaches = approaches
        self.n_repeats = n_repeats
        self.cache_root = cache_root
        self.name = name or task_type.display_name

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        method_cache: Optional["MethodResultCache"] = None,
        mission_name: str = "",
    ) -> TaskResult:
        """
        Time all approaches and return a TaskResult.

        Parameters
        ----------
        X_background : DataFrame or None
            None when no background dataset is needed (path-dependent tasks).
        method_cache : MethodResultCache | None
            If provided, cached results are used when available, and new results
            are written to the cache after measurement.
        mission_name : str
            Required when method_cache is provided to build the cache key.
        """
        params = trained_model.params.with_run_params(
            n=len(X_explain),
            m=len(X_background) if X_background is not None else 0,
        )

        approach_results: dict[str, ApproachResult] = {}
        for approach in self.approaches:
            # Try method cache first
            if method_cache is not None:
                cached = method_cache.get(approach, mission_name, self.name, params)
                if cached is not None:
                    approach_results[approach.name] = cached
                    print(f"  [approach:{approach.name}] CACHED={cached.running_time:.3f}s")
                    continue

            result = self._time_approach(approach, trained_model, X_explain, X_background, params)
            approach_results[approach.name] = result

            # Store in method cache if available and the run succeeded (not_supported and
            # memory_crash are deterministic and safe to cache; runtime_error is not).
            if method_cache is not None and not result.error:
                method_cache.put(approach, mission_name, self.name, params, result)

            if result.runtime_error:
                print(f"  [approach:{approach.name}] RUNTIME ERROR: {result.error}")
            elif result.memory_crash:
                print(f"  [approach:{approach.name}] MEMORY CRASH")
            elif result.is_estimated:
                print(
                    f"  [approach:{approach.name}] "
                    f"ESTIMATED={result.running_time:.3f}s"
                )
            else:
                print(
                    f"  [approach:{approach.name}] "
                    f"mean={result.running_time:.3f}s ± {result.std_time_s:.3f}s"
                )

        return TaskResult(task_name=self.name, params=params, approach_results=approach_results)

    # ------------------------------------------------------------------
    # Internal timing logic
    # ------------------------------------------------------------------

    def _time_approach(
        self,
        approach: Approach,
        trained_model: TrainedModel,
        X_explain: pd.DataFrame,
        X_background: Optional[pd.DataFrame],
        params: TreeParameters,
    ) -> ApproachResult:
        REPEAT_THRESHOLD_S = 10.0

        method_name = getattr(getattr(approach, "method", None), "name", "")
        task_fn = getattr(approach, self.task_type.value)
        times: list[float] = []
        any_estimated = False
        estimation_description = ""
        error: Optional[str] = None
        runtime_error = False

        try:
            first = task_fn(trained_model, X_explain, X_background)
            if first.not_supported:
                return ApproachResult(
                    approach_name=approach.name,
                    running_time=0.0,
                    std_time_s=0.0,
                    is_estimated=False,
                    error=None,
                    method=method_name,
                    not_supported=True,
                )
            if first.memory_crash:
                return ApproachResult(
                    approach_name=approach.name,
                    running_time=0.0,
                    std_time_s=0.0,
                    is_estimated=False,
                    error=None,
                    method=method_name,
                    memory_crash=True,
                )
            times.append(first.elapsed_s)
            if first.is_estimated:
                any_estimated = True
                estimation_description = first.estimation_description

            # Only repeat if the first run was fast enough.
            if first.elapsed_s < REPEAT_THRESHOLD_S:
                for _ in range(self.n_repeats - 1):
                    output = task_fn(trained_model, X_explain, X_background)
                    times.append(output.elapsed_s)
                    if output.is_estimated:
                        any_estimated = True
                        if not estimation_description:
                            estimation_description = output.estimation_description

        except Exception:
            error = traceback.format_exc()
            runtime_error = True

        if times:
            mean_t = float(np.mean(times))
            std_t = float(np.std(times))
        else:
            mean_t = std_t = 0.0

        return ApproachResult(
            approach_name=approach.name,
            running_time=mean_t,
            std_time_s=std_t,
            is_estimated=any_estimated,
            error=error,
            method=method_name,
            runtime_error=runtime_error,
            estimation_description=estimation_description,
        )
